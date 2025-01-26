from pydantic import BaseModel

class DuplicateInfo(BaseModel):
    issue_id: int
    similarity: float

class IssueData(BaseModel):
    github_id: int
    title: str
    body: str
    duplicates: list[DuplicateInfo]
