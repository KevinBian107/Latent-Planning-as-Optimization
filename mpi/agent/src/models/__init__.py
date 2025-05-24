_model_registry = {}
from agent.src.models.lpt import LatentPlannerModel
from agent.src.models.decision_transformer import DecisionTransformer
from agent.src.util_function import get_model, register_model, _model_registry

all = [get_model,register_model,_model_registry]