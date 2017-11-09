from exp_replay import ExperienceReplay

class Memories():
    def __init__(
        self,
        memory_params
        ):
        self.memories = {
            'experience-replay': ExperienceReplay(
                memory_params
            )
        }
    
    def get(self, mem):
        if mem in self.memories:
            return self.memories[mem]
        else:
            raise NotImplementedError('The memory "%s" was not found. ' 
                                        'Please try a different memory name.' % mem)