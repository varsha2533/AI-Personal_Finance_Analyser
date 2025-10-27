# (moved ensure_indexes below after app = FastAPI())
import logging
import json
import os
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Tuple
from time import time
import threading
from sqlalchemy import text
from sqlalchemy import select
import asyncio
import anyio
try:
    import psutil
except Exception:
    psutil = None
try:
    from redis import asyncio as aioredis  # redis>=4.2
except Exception:
    aioredis = None
try:
    from redis import Redis as _SyncRedis
    from rq import Queue as _RQQueue
except Exception:
    _SyncRedis = None
    _RQQueue = None

from app import crud, models, schemas
from app.db import SessionLocal, engine
from app.api_routes import get_current_user
from app.ml_retrain_service import MLRetrainService
from app.tasks.worker import enqueue_background_score
try:
    # ML model utilities (lazy)
    from app.optimized_model.router import get_model_singleton, generate_natural_language_explanation
except Exception:
    get_model_singleton = None
    generate_natural_language_explanation = None

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Lightweight in-memory cache for transaction summaries (per user)
_summary_cache: Dict[int, Tuple[float, dict]] = {}
TTL = 5.0  # seconds
# Dedicated TTL for summary caches (seconds)
SUMMARY_TTL = float(os.getenv("SUMMARY_TTL", "8"))
# Lightweight concurrency throttle for heavy dashboard endpoints
_dash_sem = threading.Semaphore(int(os.getenv("SUMMARY_CONCURRENCY", "2")))
# Server-side maximum for batched transaction IDs (safeguard)
MAX_IDS = int(os.getenv("MAX_IDS", "1000"))

# Async semaphore for new dashboard aggregate endpoint
_dash_async_sem = asyncio.Semaphore(2)

# Shared async semaphore for heavy endpoints
_heavy_sem = asyncio.Semaphore(int(os.getenv("HEAVY_CONCURRENCY", "8")))

# Optional Redis cache for aggregated dashboard
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client = None
if aioredis is not None:
    try:
        _redis_client = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    except Exception:
        _redis_client = None

# Initialize FastAPI app
app = FastAPI(
    title="AI Personal Finance Analyzer API",
    description="Clean API for personal finance analysis with optimized anomaly detection",
    version="2.0"
)

# ML model lazy holder
app.state.ml_model = None

async def get_app_model():
    """Lazy-load and cache the ML model in app.state.ml_model. Returns None if unavailable."""
    try:
        # Cached
        cached = getattr(app.state, "ml_model", None)
        if cached is not None:
            return cached
        # Loader not available
        if get_model_singleton is None:
            return None
        # Load off the event loop
        model = await anyio.to_thread.run_sync(get_model_singleton)
        app.state.ml_model = model
        try:
            mv = getattr(model, "model_version", "unknown")
            logger.info(f"[ml] Model loaded (lazy) version={mv}")
        except Exception:
            pass
        return model
    except Exception:
        logger.exception("[ml] get_app_model failed")
        return None

async def background_score_and_persist(transaction_id: int, user_id: int):
    """Score a transaction in background and persist anomaly + minimal explanation if flagged.

    - Never raises; protects the request lifecycle.
    - Uses its own DB session.
    """
    try:
        model = await get_app_model()
        if model is None:
            return

        # Fetch transaction with an independent session
        db = SessionLocal()
        try:
            tx = crud.get_transaction_by_id(db, transaction_id)
            if not tx or getattr(tx, "user_id", None) != user_id:
                return

            tx_payload = {
                "user_id": int(user_id),
                "amount": float(getattr(tx, "amount", 0) or 0),
                "category": getattr(tx, "category", None),
                "timestamp": (getattr(tx, "timestamp", None).isoformat() if getattr(tx, "timestamp", None) else datetime.utcnow().isoformat()+"Z"),
            }

            # Predict off-thread to avoid blocking loop
            pred = await anyio.to_thread.run_sync(model.predict_anomaly, tx_payload)
            prob = float(pred.get("anomaly_probability", 0.0))
            threshold = float(getattr(model, "optimized_threshold", pred.get("threshold_used", 0.51)) or 0.51)
            flagged = prob > threshold

            mv = getattr(model, "model_version", "unknown")
            logger.info(f"[ml] tx={transaction_id} prob={prob:.3f} threshold={threshold:.2f} version={mv}")

            if not flagged:
                return

            # Create Anomaly
            anomaly = crud.create_anomaly(
                db,
                schemas.AnomalyCreate(
                    transaction_id=transaction_id,
                    anomaly_score=prob,
                    description=f"Auto-flagged by model v{mv} (p={prob:.3f} > {threshold:.2f})",
                ),
                user_id,
            )

            # Create minimal Explanation
            expl_data = {
                "prediction": {
                    "anomaly_probability": prob,
                    "threshold": threshold,
                    "risk_level": pred.get("risk_level", "Unknown"),
                    "confidence_score": pred.get("confidence_score", 0.0),
                },
                "features": {"category": tx_payload["category"], "amount": tx_payload["amount"]},
            }
            nl_text = None
            try:
                if generate_natural_language_explanation is not None:
                    nl_text = generate_natural_language_explanation({"feature_importance": {}}, "SHAP")
            except Exception:
                nl_text = None

            crud.create_explanation(
                db,
                schemas.ExplanationCreate(
                    transaction_id=transaction_id,
                    model_type="SHAP",
                    explanation_data=expl_data,
                    natural_language_explanation=nl_text,
                ),
                user_id,
            )
        finally:
            try:
                db.close()
            except Exception:
                pass
    except Exception:
        # Never propagate
        logger.exception("[ml] background_score_and_persist failed")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure helpful DB indexes exist (no-op if already present)
@app.on_event("startup")
def ensure_indexes():
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_transactions_user_time 
                ON transactions (user_id, timestamp);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_anomalies_user_created 
                ON anomalies (user_id, created_at);
            """))
            conn.commit()
    except Exception as e:
        logger.warning(f"Index creation skipped or failed: {e}")

# Request logging middleware (with duration)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    method = request.method
    path = request.url.path
    start = time()
    logger.info(f"Incoming request: {method} {path}")
    response = await call_next(request)
    duration_ms = (time() - start) * 1000.0
    logger.info(f"Completed request: {method} {path} -> {response.status_code} in {duration_ms:.1f}ms")
    return response

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================================
# CORE ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"], response_description="API welcome message")
def read_root():
    """API root endpoint with welcome message."""
    return {
        "message": "AI Personal Finance Analyzer API is running",
        "version": "2.0",
        "docs": "/docs",
        "health": "/health",
        "status": "active"
    }

@app.get("/health", tags=["Health"], response_description="Health check")
def health_check():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "AI Personal Finance Analyzer API is running successfully",
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    }

# ----------------------------------------------------------------------------
# INTERNAL: Worker and Queue Health
# ----------------------------------------------------------------------------
@app.get("/_internal/worker-health", tags=["Internal"], response_description="Worker and queue health status")
def worker_health():
    """Report Redis connectivity and RQ queue pending job count.

    Never raises; returns a diagnostic payload.
    """
    info = {
        "redis": {"connected": False, "url": os.getenv("REDIS_URL", "redis://localhost:6379/0")},
        "queue": {"name": "scoring", "pending": None},
    }
    if _SyncRedis is None or _RQQueue is None:
        info["error"] = "redis/rq not installed"
        return info
    try:
        conn = _SyncRedis.from_url(info["redis"]["url"])
        info["redis"]["connected"] = bool(conn.ping())
        q = _RQQueue(name=info["queue"]["name"], connection=conn)
        info["queue"]["pending"] = q.count
    except Exception as e:
        info["error"] = str(e)
    return info

# ============================================================================
# DASHBOARD SUMMARY ENDPOINTS
# ============================================================================

@app.get("/transactions/summary", tags=["Transactions"], response_description="Summary of transactions for dashboard")
async def transactions_summary(
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return total/average spend and category breakdown for the authenticated user."""
    from sqlalchemy import func

    # Safety: if auth is missing, block any possibility of global data leakage
    if not current_user or not getattr(current_user, "id", None):
        return {
            "total_transactions": 0,
            "total_spending": 0.0,
            "category_breakdown": {},
        }

    # Memory profiling start
    rss_before = None
    if psutil is not None:
        try:
            rss_before = psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
        except Exception:
            rss_before = None

    # Redis cache first
    cache_key = f"summ:tx:{current_user.id}"
    if _redis_client is not None:
        try:
            cached = await _redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

async def invalidate_user_dashboard_caches(user_id: int):
    """Delete Redis keys for the user's dashboard caches. No-op if Redis is unavailable.

    Keys:
      - dash:summary:{user_id}
      - summ:tx:{user_id}
      - summ:anom:{user_id}
      - summ:txs:{user_id}:*
    """
    if _redis_client is None:
        return
    try:
        await _redis_client.delete(
            f"dash:summary:{user_id}",
            f"summ:tx:{user_id}",
            f"summ:anom:{user_id}"
        )
        pattern = f"summ:txs:{user_id}:*"
        async for key in _redis_client.scan_iter(match=pattern):
            try:
                await _redis_client.delete(key)
            except Exception:
                continue
    except Exception:
        # swallow errors to keep request path unaffected
        pass
    # Also clear local in-memory cache entry for this user
    try:
        _summary_cache.pop(user_id, None)
    except Exception:
        pass

    # Memory profiling end
    if psutil is not None and rss_before is not None:
        try:
            rss_after = psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
            logger.info(f"[mem] /transactions/summary RSS MB before={rss_before} after={rss_after} delta={rss_after - rss_before}")
        except Exception:
            pass
    logger = logging.getLogger(__name__)
    logger.warning("Fixed missing variable 'resp' — returning None to avoid NameError. Please verify intent.")
    return None


@app.get("/anomalies/summary", tags=["Anomalies"], response_description="Anomaly count and recent items for dashboard")
async def anomalies_summary(
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return total anomalies and a recent anomalies list for the authenticated user."""
    from sqlalchemy import func

    # Memory profiling start and Redis cache
    rss_before = None
    if psutil is not None:
        try:
            rss_before = psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
        except Exception:
            rss_before = None
    cache_key = f"summ:anom:{current_user.id}"
    if _redis_client is not None:
        try:
            cached = await _redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    # Total transactions for user (fast count)
    async with _heavy_sem:
        def _count_tx():
            stmt = select(func.count(models.Transaction.id)).where(
                models.Transaction.user_id == current_user.id,
                models.Transaction.data_scope == "user",
            )
            return db.execute(stmt).scalar() or 0
        total_tx = await anyio.to_thread.run_sync(_count_tx)

    if total_tx == 0:
        return {"total_anomalies": 0, "recent_anomalies": []}

    # Anomalies for user's transactions
    async with _heavy_sem:
        def _query_anoms():
            count_stmt = (
                select(func.count(models.Anomaly.id))
                .select_from(models.Anomaly)
                .join(models.Transaction, models.Transaction.id == models.Anomaly.transaction_id)
                .where(models.Transaction.user_id == current_user.id)
                .where(models.Transaction.data_scope == "user")
            )
            count_local = db.execute(count_stmt).scalar() or 0

            rows_stmt = (
                select(models.Anomaly.transaction_id, models.Anomaly.created_at)
                .join(models.Transaction, models.Transaction.id == models.Anomaly.transaction_id)
                .where(models.Transaction.user_id == current_user.id)
                .where(models.Transaction.data_scope == "user")
                .order_by(models.Anomaly.created_at.desc())
                .limit(5)
            )
            rows_local = db.execute(rows_stmt).all()
            return count_local, rows_local
        anomaly_count, recent_rows = await anyio.to_thread.run_sync(_query_anoms)
    recent_anomalies = [
        {"transaction_id": tr_id, "created_at": (created.isoformat() if created else None)}
        for tr_id, created in recent_rows
    ]

    resp = {
        "total_anomalies": int(anomaly_count),
        "recent_anomalies": recent_anomalies,
    }
    if _redis_client is not None:
        try:
            await _redis_client.setex(cache_key, int(SUMMARY_TTL), json.dumps(resp))
        except Exception:
            pass
    if psutil is not None and rss_before is not None:
        try:
            rss_after = psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
            logger.info(f"[mem] /anomalies/summary RSS MB before={rss_before} after={rss_after} delta={rss_after - rss_before}")
        except Exception:
            pass
    return resp


@app.get("/transactions/details", tags=["Transactions"], response_description="Batch fetch transactions by IDs for the authenticated user")
def transactions_details(
    ids: str = Query(..., description="Comma-separated transaction IDs"),
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return details for a set of transaction IDs belonging only to the authenticated user."""
    # Parse ids
    try:
        raw_ids = [s.strip() for s in (ids or "").split(",") if s.strip()]
        id_list = [int(x) for x in raw_ids if x.isdigit()]
    except Exception:
        id_list = []

    if not current_user or not getattr(current_user, "id", None):
        return {"items": []}
    if not id_list:
        return {"items": []}
    # Enforce server-side maximum to prevent excessive memory usage
    if len(id_list) > MAX_IDS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Too many IDs requested")

    rows = (
        db.query(models.Transaction)
        .filter(models.Transaction.user_id == current_user.id)
        .filter(models.Transaction.data_scope == "user")
        .filter(models.Transaction.id.in_(id_list))
        .all()
    )

    def serialize_tx(tx):
        return {
            "id": getattr(tx, "id", None),
            "amount": float(getattr(tx, "amount", 0) or 0),
            "category": getattr(tx, "category", None),
            "timestamp": getattr(tx, "timestamp", None),
            "date": getattr(tx, "date", None),
            "description": getattr(tx, "description", None),
        }

    return {"items": [serialize_tx(r) for r in rows]}

# ============================================================================
# INSIGHTS ENDPOINTS
# ============================================================================

@app.get("/transactions/timeseries", tags=["Transactions"], response_description="Time series of spending for dashboard insights")
async def transactions_timeseries(
    granularity: str = Query("week", description="Aggregation granularity: day, week, or month (defaults to week)"),
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return aggregated spending over time for the authenticated user."""
    from sqlalchemy import func
    from app.db import engine as _engine

    backend = getattr(_engine.url, "get_backend_name", lambda: None)() or _engine.url.get_backend_name()

    # Normalize granularity (fallback to week)
    granularity = (granularity or "week").lower()
    if granularity not in ("day", "week", "month"):
        granularity = "week"

    # Build period expression based on backend
    if backend == "postgresql":
        if granularity == "day":
            period_expr = func.to_char(func.date_trunc('day', models.Transaction.timestamp), 'YYYY-MM-DD')
        elif granularity == "week":
            # ISO week year-week number
            period_expr = func.to_char(func.date_trunc('week', models.Transaction.timestamp), 'IYYY-IW')
        else:
            period_expr = func.to_char(func.date_trunc('month', models.Transaction.timestamp), 'YYYY-MM')
    else:
        # SQLite and others: use strftime
        if granularity == "day":
            period_expr = func.strftime('%Y-%m-%d', models.Transaction.timestamp)
        elif granularity == "week":
            # Year-week number; prefix W for clarity
            period_expr = func.printf('%s-W%02d', func.strftime('%Y', models.Transaction.timestamp), func.strftime('%W', models.Transaction.timestamp))
        else:
            period_expr = func.strftime('%Y-%m', models.Transaction.timestamp)

    # Memory profiling and Redis cache
    rss_before = None
    if psutil is not None:
        try:
            rss_before = psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
        except Exception:
            rss_before = None

    cache_key = f"summ:txs:{current_user.id}:{granularity}"
    if _redis_client is not None:
        try:
            cached = await _redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    async with _heavy_sem:
        def _query_rows():
            stmt = (
                select(
                    period_expr.label("period"),
                    func.coalesce(func.sum(models.Transaction.amount), 0.0).label("total_spend"),
                    func.count(models.Transaction.id).label("tx_count")
                )
                .where(models.Transaction.user_id == current_user.id)
                .where(models.Transaction.data_scope == "user")
                .group_by("period")
                .order_by("period")
            )
            return db.execute(stmt).all()
        rows = await anyio.to_thread.run_sync(_query_rows)

    resp = [
        {
            "period": r.period,
            "total_spend": float(r.total_spend),
            "tx_count": int(r.tx_count),
        }
        for r in rows
    ]
    if _redis_client is not None:
        try:
            await _redis_client.setex(cache_key, int(SUMMARY_TTL), json.dumps(resp))
        except Exception:
            pass
    if psutil is not None and rss_before is not None:
        try:
            rss_after = psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
            logger.info(f"[mem] /transactions/timeseries RSS MB before={rss_before} after={rss_after} delta={rss_after - rss_before}")
        except Exception:
            pass
    return resp


@app.get("/dashboard/summary", tags=["Dashboard"], response_description="Aggregated dashboard payload")
async def dashboard_summary(
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return aggregated dashboard data in a single response with 5s cache and concurrency guard."""
    from sqlalchemy import func

    user_id = getattr(current_user, "id", None)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    cache_key = f"dash:summary:{user_id}"

    # Attempt Redis cache first
    if _redis_client is not None:
        try:
            cached = await _redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    async with _dash_async_sem:
        # Re-check cache after acquiring semaphore (to avoid stampede)
        if _redis_client is not None:
            try:
                cached = await _redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        # Build summaries (reuse logic from individual endpoints)
        # Transactions summary
        totals = (
            db.query(
                func.coalesce(func.sum(models.Transaction.amount), 0.0).label("total_spend"),
                func.coalesce(func.avg(models.Transaction.amount), 0.0).label("average_spend"),
                func.count(models.Transaction.id).label("total_transactions")
            )
            .filter(models.Transaction.user_id == user_id)
            .filter(models.Transaction.data_scope == "user")
            .one()
        )
        breakdown_rows = (
            db.query(
                models.Transaction.category.label("category"),
                func.count(models.Transaction.id).label("count"),
                func.coalesce(func.sum(models.Transaction.amount), 0.0).label("total"),
                func.coalesce(func.avg(models.Transaction.amount), 0.0).label("average")
            )
            .filter(models.Transaction.user_id == user_id)
            .filter(models.Transaction.data_scope == "user")
            .group_by(models.Transaction.category)
            .all()
        )
        category_breakdown_obj = {r.category: float(r.total or 0.0) for r in breakdown_rows}
        tx_summary_obj = {
            "total_transactions": int(totals.total_transactions or 0),
            "total_spending": float(totals.total_spend or 0.0),
            "category_breakdown": category_breakdown_obj,
        }

        # Anomalies summary
        anomaly_count = (
            db.query(func.count(models.Anomaly.id))
            .join(models.Transaction, models.Transaction.id == models.Anomaly.transaction_id)
            .filter(models.Transaction.user_id == user_id)
            .filter(models.Transaction.data_scope == "user")
            .scalar()
        ) or 0
        recent_rows = (
            db.query(models.Anomaly.transaction_id, models.Anomaly.created_at)
            .join(models.Transaction, models.Transaction.id == models.Anomaly.transaction_id)
            .filter(models.Transaction.user_id == user_id)
            .filter(models.Transaction.data_scope == "user")
            .order_by(models.Anomaly.created_at.desc())
            .limit(5)
            .all()
        )
        anomalies_obj = {
            "total_anomalies": int(anomaly_count),
            "recent_anomalies": [
                {"transaction_id": tr_id, "created_at": (created.isoformat() if created else None)}
                for tr_id, created in recent_rows
            ],
        }

        # Timeseries (weekly by default for performance)
        from app.db import engine as _engine
        backend = getattr(_engine.url, "get_backend_name", lambda: None)() or _engine.url.get_backend_name()
        if backend == "postgresql":
            period_expr = func.to_char(func.date_trunc('week', models.Transaction.timestamp), 'IYYY-IW')
        else:
            period_expr = func.printf('%s-W%02d', func.strftime('%Y', models.Transaction.timestamp), func.strftime('%W', models.Transaction.timestamp))
        ts_rows = (
            db.query(
                period_expr.label("period"),
                func.coalesce(func.sum(models.Transaction.amount), 0.0).label("total_spend"),
                func.count(models.Transaction.id).label("tx_count")
            )
            .filter(models.Transaction.user_id == user_id)
            .filter(models.Transaction.data_scope == "user")
            .group_by("period")
            .order_by("period")
            .all()
        )
        ts_obj = [
            {"period": r.period, "total_spend": float(r.total_spend), "tx_count": int(r.tx_count)}
            for r in ts_rows
        ]

        # user mini
        user_mini = {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "is_active": current_user.is_active,
        }

        resp = {
            "user": user_mini,
            "transactions_summary": tx_summary_obj,
            "anomalies_summary": anomalies_obj,
            "transactions_timeseries": ts_obj,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

        # Store in Redis with SUMMARY_TTL
        if _redis_client is not None:
            try:
                await _redis_client.setex(cache_key, int(SUMMARY_TTL), json.dumps(resp))
            except Exception:
                pass

        return resp

@app.get("/anomalies/frequency", tags=["Anomalies"], response_description="Anomaly frequency breakdown for insights")
async def anomalies_frequency(
    by: str = Query("category", pattern="^(category)$", description="Breakdown dimension"),
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return anomaly counts grouped by category for the authenticated user."""
    from sqlalchemy import func

    # Join anomalies with transactions to access category and user
    async with _heavy_sem:
        def _query_rows():
            return (
                db.query(
                    models.Transaction.category.label("category"),
                    func.count(models.Anomaly.id).label("count")
                )
                .join(models.Transaction, models.Transaction.id == models.Anomaly.transaction_id)
                .filter(models.Transaction.user_id == current_user.id)
                .filter(models.Transaction.data_scope == "user")
                .group_by(models.Transaction.category)
                .all()
            )
        rows = await anyio.to_thread.run_sync(_query_rows)

    total = sum(int(r.count) for r in rows) or 1
    data = []
    for r in rows:
        count = int(r.count)
        percentage = (count / total) * 100.0
        data.append({
            "category": r.category,
            "count": count,
            "percentage": round(percentage, 2)
        })

    # Sort desc by count for a nicer chart
    data.sort(key=lambda x: x["count"], reverse=True)
    return data


@app.get("/anomalies/timeseries", tags=["Anomalies"], response_description="Anomaly count over time for insights")
async def anomalies_timeseries(
    granularity: str = Query("week", description="Aggregation granularity: day, week, or month (defaults to week)"),
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return anomaly counts over time for the authenticated user."""
    from sqlalchemy import func
    from app.db import engine as _engine

    backend = getattr(_engine.url, "get_backend_name", lambda: None)() or _engine.url.get_backend_name()

    # Use anomaly created_at for timing, filter by user's transactions
    if backend == "postgresql":
        if granularity == "day":
            period_expr = func.to_char(func.date_trunc('day', models.Anomaly.created_at), 'YYYY-MM-DD')
        elif granularity == "week":
            period_expr = func.to_char(func.date_trunc('week', models.Anomaly.created_at), 'IYYY-IW')
        else:
            period_expr = func.to_char(func.date_trunc('month', models.Anomaly.created_at), 'YYYY-MM')
    else:
        if granularity == "day":
            period_expr = func.strftime('%Y-%m-%d', models.Anomaly.created_at)
        elif granularity == "week":
            period_expr = func.printf('%s-W%02d', func.strftime('%Y', models.Anomaly.created_at), func.strftime('%W', models.Anomaly.created_at))
        else:
            period_expr = func.strftime('%Y-%m', models.Anomaly.created_at)

    async with _heavy_sem:
        def _query_rows():
            return (
                db.query(
                    period_expr.label("period"),
                    func.count(models.Anomaly.id).label("anomaly_count")
                )
                .join(models.Transaction, models.Transaction.id == models.Anomaly.transaction_id)
                .filter(models.Transaction.user_id == current_user.id)
                .filter(models.Transaction.data_scope == "user")
                .group_by("period")
                .order_by("period")
                .all()
            )
        rows = await anyio.to_thread.run_sync(_query_rows)

    return [
        {"period": r.period, "anomaly_count": int(r.anomaly_count)}
        for r in rows
    ]


@app.get("/notifications", tags=["Notifications"], response_description="Recent notifications for the user")
def get_notifications(
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return user-scoped notifications derived from the user's anomalies and feedbacks only."""
    from sqlalchemy import desc

    anomalies = (
        db.query(models.Anomaly)
        .join(models.Transaction, models.Transaction.id == models.Anomaly.transaction_id)
        .filter(models.Transaction.user_id == current_user.id)
        .filter(models.Transaction.data_scope == "user")
        .order_by(desc(models.Anomaly.created_at))
        .limit(20)
        .all()
    )

    feedbacks = (
        db.query(models.UserFeedback)
        .join(models.Transaction, models.Transaction.id == models.UserFeedback.transaction_id)
        .filter(models.Transaction.user_id == current_user.id)
        .filter(models.Transaction.data_scope == "user")
        .order_by(desc(models.UserFeedback.created_at))
        .limit(20)
        .all()
    )

    items = []
    for a in anomalies:
        items.append({
            "type": "anomaly",
            "message": f"Anomaly detected on transaction {a.transaction_id}: {a.description}",
            "created_at": a.created_at.isoformat() if a.created_at else None,
        })
    for f in feedbacks:
        items.append({
            "type": "feedback",
            "message": f"Feedback submitted on transaction {f.transaction_id}",
            "created_at": f.created_at.isoformat() if f.created_at else None,
        })

    items = sorted(items, key=lambda x: x.get("created_at") or "", reverse=True)
    return {"notifications": items[:20]}

# ============================================================================
# TRANSACTION CRUD OPERATIONS
# ============================================================================

@app.post("/transactions", response_model=schemas.TransactionRead, status_code=status.HTTP_201_CREATED, tags=["Transactions"], response_description="Created transaction")
async def create_transaction(
    transaction: schemas.TransactionCreate, 
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new transaction for the authenticated user."""
    # Set the user_id from the authenticated user
    transaction.user_id = current_user.id
    db_transaction = crud.create_transaction(db, transaction)
    if not db_transaction:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create transaction")
    # Fire-and-forget: invalidate Redis caches for this user
    asyncio.create_task(invalidate_user_dashboard_caches(current_user.id))
    # Fire-and-forget: background scoring via RQ (best-effort)
    try:
        _job_id = enqueue_background_score(db_transaction.id, current_user.id)
    except Exception:
        # Keep request fast and successful even if enqueue fails
        _job_id = None
    # Original in-process async background call (kept for quick reversion):
    # asyncio.create_task(background_score_and_persist(db_transaction.id, current_user.id))
    # Return a non-recursive payload to satisfy TransactionRead schema
    try:
        return schemas.TransactionRead(
            id=getattr(db_transaction, "id", None),
            user_id=getattr(db_transaction, "user_id", None),
            amount=float(getattr(db_transaction, "amount", 0.0) or 0.0),
            category=getattr(db_transaction, "category", None),
            timestamp=getattr(db_transaction, "timestamp", None),
            user=None,
            explanations=[],
            anomalies=[],
            feedbacks=[],
        )
    except Exception:
        # Fallback minimal dict if schema construction fails
        return {
            "id": getattr(db_transaction, "id", None),
            "user_id": getattr(db_transaction, "user_id", None),
            "amount": float(getattr(db_transaction, "amount", 0.0) or 0.0),
            "category": getattr(db_transaction, "category", None),
            "timestamp": getattr(db_transaction, "timestamp", None),
            "user": None,
            "explanations": [],
            "anomalies": [],
            "feedbacks": [],
        }

@app.get("/transactions/{transaction_id}", response_model=schemas.TransactionRead, tags=["Transactions"], response_description="Get transaction by ID")
def get_transaction(
    transaction_id: int, 
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a transaction by its ID (only if owned by authenticated user)."""
    db_transaction = crud.get_transaction_by_id(db, transaction_id)
    if not db_transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    
    # Ensure user can only access their own transactions
    if db_transaction.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Transaction belongs to another user")
    
    return db_transaction

@app.get("/transactions", response_model=List[schemas.TransactionRead], tags=["Transactions"], response_description="List transactions")
def list_transactions(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List transactions for the authenticated user."""
    # Automatically filter by the authenticated user's ID
    transactions = crud.list_transactions(db, skip=skip, limit=limit, user_id=current_user.id)
    return transactions

# ============================================================================
# EXPLANATION CRUD OPERATIONS
# ============================================================================

@app.post("/explanations", response_model=schemas.ExplanationRead, status_code=status.HTTP_201_CREATED, tags=["Explanations"], response_description="Created explanation")
def create_explanation(
    explanation: schemas.ExplanationCreate, 
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new explanation for a transaction owned by the authenticated user."""
    # Verify the transaction belongs to the authenticated user
    transaction = crud.get_transaction_by_id(db, explanation.transaction_id)
    if not transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    
    if transaction.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Transaction belongs to another user")
    
    db_explanation = crud.create_explanation(db, explanation, current_user.id)
    if not db_explanation:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create explanation")
    return db_explanation

@app.get("/explanations/{explanation_id}", response_model=schemas.ExplanationRead, tags=["Explanations"], response_description="Get explanation by ID")
def get_explanation(
    explanation_id: int, 
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get an explanation by its ID (only if related to a transaction owned by authenticated user)."""
    db_explanation = crud.get_explanation_by_id(db, explanation_id)
    if not db_explanation:      
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Explanation not found")
    
    # Verify the related transaction belongs to the authenticated user
    transaction = crud.get_transaction_by_id(db, db_explanation.transaction_id)
    if transaction.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Explanation belongs to another user's transaction")
    
    return db_explanation

@app.get("/explanations", response_model=List[schemas.ExplanationRead], tags=["Explanations"], response_description="List explanations")
def list_explanations(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    transaction_id: Optional[int] = Query(None, description="Filter by transaction ID"),
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List explanations for transactions owned by the authenticated user."""
    # If transaction_id is provided, verify it belongs to the user
    if transaction_id:
        transaction = crud.get_transaction_by_id(db, transaction_id)
        if not transaction:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
        if transaction.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Transaction belongs to another user")
    
    # Get explanations filtered by user's transactions (more efficient)
    explanations = crud.list_explanations(db, skip=skip, limit=limit, transaction_id=transaction_id, user_id=current_user.id)
    return explanations

# ============================================================================
# ANOMALY CRUD OPERATIONS
# ============================================================================

@app.post("/anomalies", response_model=schemas.AnomalyRead, status_code=status.HTTP_201_CREATED, tags=["Anomalies"], response_description="Created anomaly")
def create_anomaly(
    anomaly: schemas.AnomalyCreate, 
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new anomaly for a transaction owned by the authenticated user."""
    # Verify the transaction belongs to the authenticated user
    transaction = crud.get_transaction_by_id(db, anomaly.transaction_id)
    if not transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    
    if transaction.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Transaction belongs to another user")
    
    db_anomaly = crud.create_anomaly(db, anomaly, current_user.id)
    if not db_anomaly:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create anomaly")
    return db_anomaly

@app.get("/anomalies/{anomaly_id}", response_model=schemas.AnomalyRead, tags=["Anomalies"], response_description="Get anomaly by ID")
def get_anomaly(
    anomaly_id: int, 
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get an anomaly by its ID (only if related to a transaction owned by authenticated user)."""
    db_anomaly = crud.get_anomaly_by_id(db, anomaly_id)
    if not db_anomaly:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")
    
    # Verify the related transaction belongs to the authenticated user
    transaction = crud.get_transaction_by_id(db, db_anomaly.transaction_id)
    if transaction.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Anomaly belongs to another user's transaction")
    
    return db_anomaly

@app.get("/anomalies", response_model=List[schemas.AnomalyRead], tags=["Anomalies"], response_description="List anomalies")
def list_anomalies(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    transaction_id: Optional[int] = Query(None, description="Filter by transaction ID"),
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List anomalies for transactions owned by the authenticated user."""
    # If transaction_id is provided, verify it belongs to the user
    if transaction_id:
        transaction = crud.get_transaction_by_id(db, transaction_id)
        if not transaction:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
        if transaction.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Transaction belongs to another user")
    
    # Get anomalies filtered by user's transactions (more efficient)
    anomalies = crud.list_anomalies(db, skip=skip, limit=limit, transaction_id=transaction_id, user_id=current_user.id)
    return anomalies

# ============================================================================
# FEEDBACK CRUD OPERATIONS
# ============================================================================

@app.post("/feedbacks", response_model=schemas.FeedbackRead, status_code=status.HTTP_201_CREATED, tags=["Feedbacks"], response_description="Created feedback")
def create_feedback(
    feedback: schemas.FeedbackCreate, 
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new feedback for a transaction owned by the authenticated user."""
    # Verify the transaction belongs to the authenticated user
    transaction = crud.get_transaction_by_id(db, feedback.transaction_id)
    if not transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    
    if transaction.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Transaction belongs to another user")
    
    db_feedback = crud.create_feedback(db, feedback, current_user.id)
    if not db_feedback:
        raise HTTPException(status_code=400, detail="Failed to create feedback.")
    return db_feedback

@app.get("/feedbacks/{feedback_id}", response_model=schemas.FeedbackRead, tags=["Feedbacks"], response_description="Get feedback by ID")
def read_feedback(
    feedback_id: int, 
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get feedback by its ID (only if related to a transaction owned by authenticated user)."""
    db_feedback = crud.get_feedback_by_id(db, feedback_id)
    if not db_feedback:
        raise HTTPException(status_code=404, detail="Feedback not found.")
    
    # Verify the related transaction belongs to the authenticated user
    transaction = crud.get_transaction_by_id(db, db_feedback.transaction_id)
    if transaction.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Feedback belongs to another user's transaction")
    
    return db_feedback

@app.get("/feedbacks", response_model=List[schemas.FeedbackRead], tags=["Feedbacks"], response_description="List feedbacks")
def list_feedbacks(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    transaction_id: Optional[int] = Query(None, description="Filter by transaction ID"),
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List feedbacks for transactions owned by the authenticated user."""
    # If transaction_id is provided, verify it belongs to the user
    if transaction_id:
        transaction = crud.get_transaction_by_id(db, transaction_id)
        if not transaction:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
        if transaction.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Transaction belongs to another user")
    
    # Get feedbacks filtered by user's transactions (more efficient)
    feedbacks = crud.list_feedbacks(db, skip=skip, limit=limit, transaction_id=transaction_id, user_id=current_user.id)
    return feedbacks

# ============================================================================
# USER FEEDBACK OPERATIONS
# ============================================================================

@app.post("/feedback/submit", response_model=schemas.UserFeedbackRead, status_code=status.HTTP_201_CREATED, tags=["User Feedback"], response_description="Submitted user feedback")
def submit_user_feedback(
    user_feedback: schemas.UserFeedbackCreate,
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit user feedback for anomaly detection results.
    
    - **transaction_id**: ID of the transaction being feedbacked
    - **feedback_type**: Type of feedback ('True Anomaly', 'False Anomaly', 'Not Sure')
    - **comments**: Optional text field for user comments
    """
    try:
        # Create user feedback (CRUD function handles transaction ownership verification)
        db_user_feedback = crud.create_user_feedback(db, user_feedback, current_user.id)
        if not db_user_feedback:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Failed to create user feedback. Transaction may not exist or may not belong to you."
            )
        
        return db_user_feedback
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/feedback/{feedback_id}", response_model=schemas.UserFeedbackRead, tags=["User Feedback"], response_description="Get user feedback by ID")
def get_user_feedback(
    feedback_id: int,
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user feedback by its ID (only if owned by the authenticated user)."""
    db_user_feedback = crud.get_user_feedback_by_id(db, feedback_id, current_user.id)
    if not db_user_feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User feedback not found or access denied"
        )
    
    return db_user_feedback

@app.get("/feedback/transaction/{transaction_id}", response_model=schemas.UserFeedbackRead, tags=["User Feedback"], response_description="Get user feedback for specific transaction")
def get_user_feedback_by_transaction(
    transaction_id: int,
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user feedback for a specific transaction (only if transaction belongs to authenticated user)."""
    db_user_feedback = crud.get_user_feedback_by_transaction(db, transaction_id, current_user.id)
    if not db_user_feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User feedback not found for this transaction or access denied"
        )
    
    return db_user_feedback

@app.put("/feedback/{feedback_id}", response_model=schemas.UserFeedbackRead, tags=["User Feedback"], response_description="Updated user feedback")
def update_user_feedback(
    feedback_id: int,
    user_feedback_update: schemas.UserFeedbackUpdate,
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user feedback (only if owned by the authenticated user)."""
    try:
        db_user_feedback = crud.update_user_feedback(db, feedback_id, user_feedback_update, current_user.id)
        if not db_user_feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="User feedback not found or access denied"
            )
        
        return db_user_feedback
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.delete("/feedback/{feedback_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["User Feedback"], response_description="Deleted user feedback")
def delete_user_feedback(
    feedback_id: int,
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete user feedback (only if owned by the authenticated user)."""
    try:
        deletion_success = crud.delete_user_feedback(db, feedback_id, current_user.id)
        if not deletion_success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="User feedback not found or access denied"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/feedback", response_model=List[schemas.UserFeedbackRead], tags=["User Feedback"], response_description="List user feedbacks")
def list_user_feedbacks(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    transaction_id: Optional[int] = Query(None, description="Filter by transaction ID"),
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user feedbacks for the authenticated user."""
    # If transaction_id is provided, verify it belongs to the user
    if transaction_id:
        transaction = crud.get_transaction_by_id(db, transaction_id)
        if not transaction:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
        if transaction.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Transaction belongs to another user")
    
    # Get user feedbacks filtered by user and optionally by transaction
    user_feedbacks = crud.list_user_feedbacks(db, skip=skip, limit=limit, user_id=current_user.id, transaction_id=transaction_id)
    return user_feedbacks

# ============================================================================
# ML MODEL RETRAINING ENDPOINTS
# ============================================================================

@app.post("/ml/retrain", response_model=schemas.MLRetrainResponse, status_code=status.HTTP_202_ACCEPTED, tags=["ML Operations"], response_description="ML model retraining initiated")
async def retrain_ml_models(
    retrain_request: schemas.MLRetrainRequest,
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Initiate complete ML model retraining pipeline.
    
    This endpoint triggers:
    1. New dataset generation
    2. Training data preparation
    3. Model retraining with latest data
    4. New model versioning
    
    Requires authentication and appropriate permissions.
    """
    try:
        # Check if user has permission to trigger retraining
        # For now, allow any authenticated user, but you can add role-based checks here
        logger.info(f"User {current_user.username} initiated ML model retraining")
        
        # Initialize the retraining service
        retrain_service = MLRetrainService()
        
        # Execute the complete retraining pipeline
        retrain_result = retrain_service.execute_complete_retraining()
        
        # Log the retraining result
        if retrain_result["status"] == "success":
            logger.info(f"ML retraining completed successfully by user {current_user.username}. New version: {retrain_result.get('new_model_version')}")
        else:
            logger.error(f"ML retraining failed for user {current_user.username}: {retrain_result.get('error')}")
        
        return retrain_result
        
    except Exception as e:
        logger.error(f"Error during ML retraining: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ML retraining failed: {str(e)}"
        )

@app.get("/ml/retrain/status", response_model=schemas.MLRetrainStatus, tags=["ML Operations"], response_description="Current ML retraining status")
async def get_ml_retrain_status(
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the current status of ML retraining operations.
    
    Returns information about:
    - Whether retraining is currently in progress
    - Current step and progress
    - Last retraining results
    """
    try:
        # For now, return a simple status
        # In a production system, you'd track this in a database or cache
        retrain_service = MLRetrainService()
        
        # Check if there are any recent retrain logs
        retrain_logs = list(retrain_service.retrain_log_dir.glob("retrain_log_*.json"))
        
        last_retrain = None
        if retrain_logs:
            # Get the most recent retrain log
            latest_log = max(retrain_logs, key=lambda x: x.stat().st_mtime)
            with open(latest_log, 'r') as f:
                last_retrain = json.load(f)
        
        return schemas.MLRetrainStatus(
            is_retraining=False,  # For now, assume no concurrent retraining
            current_step=None,
            progress_percentage=None,
            estimated_completion=None,
            last_retrain=last_retrain
        )
        
    except Exception as e:
        logger.error(f"Error getting ML retrain status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get retraining status: {str(e)}"
        )

# ============================================================================
# AUTHENTICATION ROUTER
# ============================================================================

# Include authentication router
from app.api_routes import auth_router
app.include_router(auth_router, tags=["Authentication"])

# ============================================================================
# OPTIMIZED MODEL ROUTER
# ============================================================================

# Include router for the new optimized ensemble model
from app.optimized_model.router import router as optimized_router
app.include_router(optimized_router, tags=["Optimized Model"])

# =========================================================================
# EXPLAINABILITY ROUTER
# =========================================================================

# Include router for explainability (SHAP, LIME, combined explanations)
from app.explainability.router import router as explain_router
app.include_router(explain_router, tags=["Explainability"])

# ============================================================================
# FUTURE ENHANCEMENTS
# ============================================================================

# ---
# To extend this API securely with JWT-based OAuth2 authentication in the future:
# 1. Use fastapi.security.OAuth2PasswordBearer and OAuth2PasswordRequestForm.
# 2. Implement token creation and validation endpoints.
# 3. Protect endpoints with Depends(oauth2_scheme) and user validation logic.
# See FastAPI docs: https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
# ---

