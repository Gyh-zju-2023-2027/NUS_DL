import re
from typing import Tuple, Optional, List, Union, Literal, Set, Dict, get_args
from pydantic import BaseModel
from models.src.setting import USE_SENSOR_PROCESS
from models.src.new_key_mapping import AllowedSuggestionType, AllowedSuggestionKey

pose_keypoints_with_official_name = [
    {"id": 0, "name": "Nose"},  
    {"id": 1, "name": "Left eye inner"},  
    {"id": 2, "name": "Left eye"},  
    {"id": 3, "name": "Left eye outer"},  
    {"id": 4, "name": "Right eye inner"},  
    {"id": 5, "name": "Right eye"},  
    {"id": 6, "name": "Right eye outer"},  
    {"id": 7, "name": "Left ear"},  
    {"id": 8, "name": "Right ear"},  
    {"id": 9, "name": "Mouth left"},  
    {"id": 10, "name": "Mouth right"},  
    {"id": 11, "name": "Left shoulder"},  
    {"id": 12, "name": "Right shoulder"},  
    {"id": 13, "name": "Left elbow"},  
    {"id": 14, "name": "Right elbow"},  
    {"id": 15, "name": "Left wrist"},  
    {"id": 16, "name": "Right wrist"},  
    {"id": 17, "name": "Left pinky"},  
    {"id": 18, "name": "Right pinky"},  
    {"id": 19, "name": "Left index"},  
    {"id": 20, "name": "Right index"},  
    {"id": 21, "name": "Left thumb"},  
    {"id": 22, "name": "Right thumb"},  
    {"id": 23, "name": "Left hip"},  
    {"id": 24, "name": "Right hip"},  
    {"id": 25, "name": "Left knee"},  
    {"id": 26, "name": "Right knee"},  
    {"id": 27, "name": "Left ankle"},  
    {"id": 28, "name": "Right ankle"},  
    {"id": 29, "name": "Left heel"},  
    {"id": 30, "name": "Right heel"},  
    {"id": 31, "name": "Left foot index"},  
    {"id": 32, "name": "Right foot index"},  
]

PREDEFINE_pose_dim_keys = [point["name"].lower().replace(' ', '_') for point in pose_keypoints_with_official_name]

# 只保留pose相关的数据键
all_data_keys = PREDEFINE_pose_dim_keys

class ParsedSuggestion(BaseModel):
    key: AllowedSuggestionKey
    suggestion_type: AllowedSuggestionType
    description: str

class StrokeExpertSuggestion(BaseModel):
    stroke_id: int
    suggestion: ParsedSuggestion

class RoundExpertSuggestions(BaseModel):
    round_meta_info: str
    stroke_suggestions: List[StrokeExpertSuggestion]
    summary_suggestions: List[ParsedSuggestion]

class RoundDataIncludesPoseSensor(BaseModel):
    round_meta_info: str
    pose_data: List
    stroke_mask: List[int]
    expert_suggestions: List[RoundExpertSuggestions]
    expert_sugg_key_id_set: Set[int]




