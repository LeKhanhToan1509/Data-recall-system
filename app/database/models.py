from sqlalchemy import create_engine, ForeignKey, Boolean, Column, Integer, String, DateTime, Float #type: ignore
from sqlalchemy.sql import func #type: ignore
from sqlalchemy.dialects.postgresql import JSONB, UUID #type: ignore
from sqlalchemy.orm import relationship, declarative_base, sessionmaker #type: ignore
from database.database import Base
from configure import Config
import uuid

class Image(Base):
    __tablename__ = 'image'
    __table_args__ = {'extend_existing': True}
    id = Column('Id_Image', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column('URL', String)
    description = Column('Description', String)
    meta_data = Column('Meta_data', JSONB, nullable=True)  # Sửa tên cột và thêm nullable=True nếu cần
    metric = Column('Metric', JSONB, nullable=True)  # Sửa tên cột và thêm nullable=True nếu cần

    objects = relationship('Object', back_populates='image')

class Object(Base):
    __tablename__ = 'object'
    __table_args__ = {'extend_existing': True}
    id = Column('Id_Object', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column('Image_Id', UUID(as_uuid=True), ForeignKey('image.Id_Image'),default=uuid.uuid4)  # Khóa ngoại tham chiếu đến bảng Image
    segment = Column('Segment', JSONB, nullable=True)
    bbox = Column('Bbox', JSONB, nullable=True)
    class_name = Column('Class_Name', String)
    
    image = relationship('Image', back_populates='objects')