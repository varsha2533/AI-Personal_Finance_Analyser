from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional, Type, TypeVar, Any
from app import models, schemas

T = TypeVar('T')


def create_object(db: Session, model: Type[T], obj_in: Any) -> Optional[T]:
    """Generic helper to create a DB object from a Pydantic schema."""
    try:
        db_obj = model(**obj_in.dict())
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error creating {model.__name__}: {e}")
        return None

def create_transaction(db: Session, transaction: schemas.TransactionCreate) -> Optional[models.Transaction]:
    """Create a new transaction."""
    return create_object(db, models.Transaction, transaction)


def get_transaction_by_id(db: Session, transaction_id: int) -> Optional[models.Transaction]:
    """Get a transaction by its ID."""
    return db.query(models.Transaction).filter(models.Transaction.id == transaction_id).first()


def list_transactions(db: Session, skip: int = 0, limit: int = 100, user_id: Optional[int] = None) -> List[models.Transaction]:
    """List transactions, optionally filtered by user_id."""
    query = db.query(models.Transaction)
    if user_id is not None:
        query = query.filter(models.Transaction.user_id == user_id)
    return query.offset(skip).limit(limit).all()


def create_explanation(db: Session, explanation: schemas.ExplanationCreate) -> Optional[models.Explanation]:
    """Create a new explanation."""
    return create_object(db, models.Explanation, explanation)


def get_explanation_by_id(db: Session, explanation_id: int) -> Optional[models.Explanation]:
    """Get an explanation by its ID."""
    return db.query(models.Explanation).filter(models.Explanation.id == explanation_id).first()


def list_explanations(db: Session, skip: int = 0, limit: int = 100, transaction_id: Optional[int] = None) -> List[models.Explanation]:
    """List explanations, optionally filtered by transaction_id."""
    query = db.query(models.Explanation)
    if transaction_id is not None:
        query = query.filter(models.Explanation.transaction_id == transaction_id)
    return query.offset(skip).limit(limit).all()


def create_anomaly(db: Session, anomaly: schemas.AnomalyCreate) -> Optional[models.Anomaly]:
    """Create a new anomaly."""
    return create_object(db, models.Anomaly, anomaly)


def get_anomaly_by_id(db: Session, anomaly_id: int) -> Optional[models.Anomaly]:
    """Get an anomaly by its ID."""
    return db.query(models.Anomaly).filter(models.Anomaly.id == anomaly_id).first()


def list_anomalies(db: Session, skip: int = 0, limit: int = 100, transaction_id: Optional[int] = None) -> List[models.Anomaly]:
    """List anomalies, optionally filtered by transaction_id."""
    query = db.query(models.Anomaly)
    if transaction_id is not None:
        query = query.filter(models.Anomaly.transaction_id == transaction_id)
    return query.offset(skip).limit(limit).all()


def create_feedback(db: Session, feedback: schemas.FeedbackCreate) -> Optional[models.Feedback]:
    """Create a new feedback."""
    return create_object(db, models.Feedback, feedback)


def get_feedback_by_id(db: Session, feedback_id: int) -> Optional[models.Feedback]:
    """Get feedback by its ID."""
    return db.query(models.Feedback).filter(models.Feedback.id == feedback_id).first()


def list_feedbacks(db: Session, skip: int = 0, limit: int = 100, transaction_id: Optional[int] = None) -> List[models.Feedback]:
    """List feedbacks, optionally filtered by transaction_id."""
    query = db.query(models.Feedback)
    if transaction_id is not None:
        query = query.filter(models.Feedback.transaction_id == transaction_id)
    return query.offset(skip).limit(limit).all()