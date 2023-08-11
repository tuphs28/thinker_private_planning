from thinker.log import SLogWorker
from thinker import util

if __name__ == "__main__":
    flags = util.parse(override=True)
    log_worker = SLogWorker(flags)
    log_worker.start()
