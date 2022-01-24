import snet.infra as db
import datetime

from snet.fenv import  ENV
from snet.flow import FLOWGEN
#--------------------------------------------------
# CORE PARAMETERS----------------------------------
#--------------------------------------------------
AUTOSAVE =              True                #HINT [saves figures]
dbinfra =               db.infra_1          #HINT

APP_PROFILES = {

    'app_1':    {'pro_1': (4, [1,2,1]), 'pro_2': (5, [2,2,1]), 'pro_3': (6, [3,2,1]) },
    'app_2':    {'pro_1': (5, [1,2,2]), 'pro_2': (6, [2,3,1]), 'pro_3': (7, [3,3,1]) },
    'app_3':    {'pro_1': (4, [2,1,1]), 'pro_2': (5, [2,1,2]), 'pro_3': (6, [3,1,2]) },
    'app_4':    {'pro_1': (7, [3,3,1]), 'pro_2': (6, [2,2,2]), 'pro_3': (6, [1,1,4]) },
    'app_5':    {'pro_1': (7, [1,2,4]), 'pro_2': (7, [2,4,1]), 'pro_3': (6, [4,1,1]) },

}
MAX_N_TASKS =           7                   #HINT
G_LAYERS =              [256,256,256]       #HINT: DQN Arch
G_ENV_ACTION =          dbinfra().A
G_ENV_SHAPE =           (MAX_N_TASKS+3)*(MAX_N_TASKS+1) + (MAX_N_TASKS+1) 
#--------------------------------------------------

def stamp_now(format="%Y_%m_%d_%H_%M_%S"):
    return datetime.datetime.strftime(datetime.datetime.now(), format)


#--------------------#--------------------
# Environment Definition
#--------------------#--------------------

def get_env(infra, flow_generator, name="", initial_reset=False):
    return ENV   (  arg_infra =      infra, 
                    flow_generator = flow_generator, 
                    state_vector_len = G_ENV_SHAPE,
                    single_gene =    False,             #< [soultion dynamics]
                    name=            name, 
                    initial_reset =  initial_reset,     #<---- call at explorer initialization
                )



class aproxy:
    def __init__(self) -> None:
        pass
        
    def add_argument(self, arg_word, type=int, default=0, help=''):
        setattr(self, arg_word[2:], default)

    def parse_args(self):
        return self



