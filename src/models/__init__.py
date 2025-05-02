_model_registry = {}
from src.models.LPT import LatentPlannerModel
from src.models.decision_transformer import DecisionTransformer
from src.util_function import get_model,register_model,_model_registry

all = [get_model,register_model,_model_registry]