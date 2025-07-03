import logging
from fastapi import FastAPI, Depends, HTTPException, status, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

from app import crud, models, schemas
from app.db import SessionLocal, engine

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="API for detecting anomalies in transactions",
    version="1.0"
)

# CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health", tags=["Health"], response_description="Health check")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/transactions", response_model=schemas.TransactionRead, status_code=status.HTTP_201_CREATED, tags=["Transactions"], response_description="Created transaction")
def create_transaction(transaction: schemas.TransactionCreate, db: Session = Depends(get_db)):
    """Create a new transaction."""
    db_transaction = crud.create_transaction(db, transaction)
    if not db_transaction:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create transaction")
    return db_transaction

@app.get("/transactions/{transaction_id}", response_model=schemas.TransactionRead, tags=["Transactions"], response_description="Get transaction by ID")
def get_transaction(transaction_id: int, db: Session = Depends(get_db)):
    """Get a transaction by its ID."""
    db_transaction = crud.get_transaction_by_id(db, transaction_id)
    if not db_transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    return db_transaction

@app.get("/transactions", response_model=List[schemas.TransactionRead], tags=["Transactions"], response_description="List transactions")
def list_transactions(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    db: Session = Depends(get_db)
):
    """List transactions, optionally filtered by user ID."""
    transactions = crud.list_transactions(db, skip=skip, limit=limit, user_id=user_id)
    return transactions

@app.post("/explanations", response_model=schemas.ExplanationRead, status_code=status.HTTP_201_CREATED, tags=["Explanations"], response_description="Created explanation")
def create_explanation(explanation: schemas.ExplanationCreate, db: Session = Depends(get_db)):
    """Create a new explanation."""
    db_explanation = crud.create_explanation(db, explanation)
    if not db_explanation:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create explanation")
    return db_explanation

@app.get("/explanations/{explanation_id}", response_model=schemas.ExplanationRead, tags=["Explanations"], response_description="Get explanation by ID")
def get_explanation(explanation_id: int, db: Session = Depends(get_db)):
    """Get an explanation by its ID."""
    db_explanation = crud.get_explanation_by_id(db, explanation_id)
    if not db_explanation:      
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Explanation not found")
    return db_explanation

@app.get("/explanations", response_model=List[schemas.ExplanationRead], tags=["Explanations"], response_description="List explanations")
def list_explanations(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    transaction_id: Optional[int] = Query(None, description="Filter by transaction ID"),
    db: Session = Depends(get_db)
):
    """List explanations, optionally filtered by transaction ID."""
    explanations = crud.list_explanations(db, skip=skip, limit=limit, transaction_id=transaction_id)
    return explanations

@app.post("/anomalies", response_model=schemas.AnomalyRead, status_code=status.HTTP_201_CREATED, tags=["Anomalies"], response_description="Created anomaly")
def create_anomaly(anomaly: schemas.AnomalyCreate, db: Session = Depends(get_db)):
    """Create a new anomaly."""
    db_anomaly = crud.create_anomaly(db, anomaly)
    if not db_anomaly:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create anomaly")
    return db_anomaly

@app.get("/anomalies/{anomaly_id}", response_model=schemas.AnomalyRead, tags=["Anomalies"], response_description="Get anomaly by ID")
def get_anomaly(anomaly_id: int, db: Session = Depends(get_db)):
    """Get an anomaly by its ID."""
    db_anomaly = crud.get_anomaly_by_id(db, anomaly_id)
    if not db_anomaly:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")
    return db_anomaly

@app.get("/anomalies", response_model=List[schemas.AnomalyRead], tags=["Anomalies"], response_description="List anomalies")
def list_anomalies(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    transaction_id: Optional[int] = Query(None, description="Filter by transaction ID"),
    db: Session = Depends(get_db)
):
    """List anomalies, optionally filtered by transaction ID."""
    anomalies = crud.list_anomalies(db, skip=skip, limit=limit, transaction_id=transaction_id)
    return anomalies

@app.post("/feedbacks", response_model=schemas.FeedbackRead, status_code=status.HTTP_201_CREATED, tags=["Feedbacks"], response_description="Created feedback")
def create_feedback(feedback: schemas.FeedbackCreate, db: Session = Depends(get_db)):
    """Create a new feedback."""
    db_feedback = crud.create_feedback(db, feedback)
    if not db_feedback:
        raise HTTPException(status_code=400, detail="Failed to create feedback.")
    return db_feedback

@app.get("/feedbacks/{feedback_id}", response_model=schemas.FeedbackRead, tags=["Feedbacks"], response_description="Get feedback by ID")
def read_feedback(feedback_id: int, db: Session = Depends(get_db)):
    """Get feedback by its ID."""
    db_feedback = crud.get_feedback_by_id(db, feedback_id)
    if not db_feedback:
        raise HTTPException(status_code=404, detail="Feedback not found.")
    return db_feedback

@app.get("/feedbacks", response_model=List[schemas.FeedbackRead], tags=["Feedbacks"], response_description="List feedbacks")
def list_feedbacks(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    transaction_id: Optional[int] = Query(None, description="Filter by transaction ID"),
    db: Session = Depends(get_db)
):
    """List feedbacks, optionally filtered by transaction ID."""
    feedbacks = crud.list_feedbacks(db, skip=skip, limit=limit, transaction_id=transaction_id)
    return feedbacks

@app.get("/", tags=["Root"], response_description="API welcome message")
def read_root():
    
    return {
        "message": "Working"
    }

# ---
# To extend this API securely with JWT-based OAuth2 authentication in the future:
# 1. Use fastapi.security.OAuth2PasswordBearer and OAuth2PasswordRequestForm.
# 2. Implement token creation and validation endpoints.
# 3. Protect endpoints with Depends(oauth2_scheme) and user validation logic.
# See FastAPI docs: https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/

