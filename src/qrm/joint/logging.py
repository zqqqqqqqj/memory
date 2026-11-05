import json
import os
from pathlib import Path
import time

def write_training_status(out, status, step, **extra):
    path = Path(out)/'training.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(status=status, step=step, pid=os.getpid(),
                                         updated_unix=time.time(), **extra))+'\n')
    temporary.replace(path)
