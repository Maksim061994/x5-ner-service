import multiprocessing
import os

bind = "0.0.0.0:8000"
cpu_count = multiprocessing.cpu_count()

workers = min(2, max(1, cpu_count // 2))
worker_connections = 1000
threads = max(2, cpu_count // workers)
max_requests = 1000
max_requests_jitter = 100

# -t
timeout = 900

preload_app = True
worker_class = "uvicorn.workers.UvicornWorker"

keepalive = 5
max_requests_jitter = 50
worker_tmp_dir = "/dev/shm"

# Настройки логирования
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'


def when_ready(server):
    """Вызывается когда сервер готов принимать соединения"""
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    """Вызывается при получении SIGINT/SIGTERM воркером"""
    worker.log.info("worker received INT or TERM signal")

def pre_fork(server, worker):
    """Вызывается перед форком воркера"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Вызывается после форка воркера"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)
    
    # Настройка для ML моделей
    import os
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
