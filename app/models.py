from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db import Base

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    explanations = relationship("Explanation", back_populates="transaction", cascade="all, delete-orphan")
    anomalies = relationship("Anomaly", back_populates="transaction", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="transaction", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Transaction(id={self.id}, user_id={self.user_id}, amount={self.amount}, category='{self.category}', timestamp={self.timestamp})>"

class Anomaly(Base):
    __tablename__ = "anomalies"
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), index=True, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    description = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    transaction = relationship("Transaction", back_populates="anomalies")

    def __repr__(self):
        return f"<Anomaly(id={self.id}, transaction_id={self.transaction_id}, anomaly_score={self.anomaly_score}, description='{self.description}')>"

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), index=True, nullable=False)
    feedback_text = Column(String, nullable=False)
    is_anomaly_correct = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    transaction = relationship("Transaction", back_populates="feedbacks")

    def __repr__(self):
        return f"<Feedback(id={self.id}, transaction_id={self.transaction_id}, is_anomaly_correct={self.is_anomaly_correct})>"

class Explanation(Base):
    __tablename__ = "explanations"
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), index=True, nullable=False)
    method = Column(String, nullable=False)
    explanation_data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    transaction = relationship("Transaction", back_populates="explanations")

    def __repr__(self):
        return f"<Explanation(id={self.id}, transaction_id={self.transaction_id}, method='{self.method}')>"
