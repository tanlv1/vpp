from sqlalchemy import Column, Integer, String, Float
from app.models.database import Base


class ObjectMetadata(Base):
    __tablename__ = "object_metadata"

    id = Column(Integer, primary_key=True, index=True)
    object_name = Column(String, index=True)
    confidence = Column(Float)
    video_id = Column(Integer)

    # Additional fields as necessary
