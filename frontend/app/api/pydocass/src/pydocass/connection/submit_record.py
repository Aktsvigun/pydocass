from typing import Literal
from sqlalchemy.exc import OperationalError
import time

from .database import structures, get_db


# def submit_record_with_retry(session, record, retries=3, delay=2):
#     for attempt in range(retries):
#         try:
#             session.add(record)
#             session.commit()
#             return
#         except OperationalError:
#             session.rollback()
#             if attempt < retries - 1:
#                 time.sleep(delay)
#             else:
#                 raise


def submit_record(
    table: Literal["responses", "feedback", "inputs"],
    retries: int = 3,
    delay: int = 2,
    **kwargs
):
    for attempt in range(retries):
        try:
            record = structures[table](**kwargs)
            with get_db() as db:
                db.add(record)
                db.commit()
                db.refresh(record)
        except OperationalError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise
