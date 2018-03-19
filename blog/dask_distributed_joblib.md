# Tutorial: How to use `dask-distributed` to manage a pool of workers on multiple machines, and use them in `joblib`

In parallel computing, an embarrassingly parallel problem is one which is obviously decomposable into many identical but separate subtasks. For such tasks, `joblib` is a very easy-to-use Python package, which allows to distribute work on multiple procesors. It is used for instance internally in `scikit-learn` for parallel grid search and cross-validation. `joblib` makes parallel computing ridiculously easy ([see the doc](https://pythonhosted.org/joblib/index.html)):

```py
from joblib import Parallel, delayed
from math import sqrt
result = Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
# result = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
```
However, `joblib` is limited to local processors, which means it is easy to make full use of an 80-core cluster, but it is more complicated to distribute tasks on (for example) the 40 desktops of an university lab room, on which you might have a direct SSH access.


But recently, `dask-distributed` implemented a `joblib` backend, which makes it very easy to use if you are familiar with `joblib`.
The cool part is that your python script will (almost) not change.
Here are the details.

## Install
First of all, you will need to install the following packages:

```console
conda install dask distributed -c conda-forge
conda install bokeh
pip install paramiko joblib
```

- `dask` is a flexible parallel computing library.
- `dask.distributed` is a lightweight library for distributed computing.
- `bokeh` is an interactive visualization library.
- `paramiko` is an implementation of the SSHv2 protocol.
- `joblib` is a set of tools to provide lightweight pipelining.

## How to deal with passwords in `dask-ssh`

In short, `dask-ssh` is the command you need, and it is available after installing `dask-distributed`. However, the connection to the servers may require a password, which you don't want to type every time you start your script, and definetely not for each one of the servers. Here is one way to handle this issue:

1. First, you need a ssh key. Check if there is already a ssh key (called e.g. `id_rsa` and `id_rsa.pub`) in your machine:

  ```console
  ls -a ~/.ssh
  ```

2. If there is a ssh key and you know its pass-phrase, use it. Otherwise, create a ssh key with:

  ```console
  ssh-keygen
  ```

3. Then add your public key to all your distant servers (e.g. for me `lame10` and `lame11` with my username `tdupre`):

  ```console
  ssh-copy-id tdupre@lame10
  ssh-copy-id tdupre@lame11
  ```

4. Start an ssh-agent in the background:

  ```console
  eval "$(ssh-agent -s)"
  ```

5. Add your private key to the ssh-agent:

  ```console
  ssh-add ~/.ssh/id_rsa
  ```

6. Test that the connection to your server is now password-free:

  ```console
  ssh tdupre@lame10
  ```

7. Tadaaa ! You will have to repete steps 4 and 5 in every terminal in which you want to run `dask-ssh`.

## How to create the scheduler and the workers in each server

The scheduler is the process which receives the work from `joblib`, and dispatches it to the workers. The workers are the processes on the distant servers which are going to perform the tasks.
To create the scheduler and the workers, all you need is `dask-ssh`:

```console
dask-ssh \
    --scheduler localhost \
    --nprocs 1 \
    --nthreads 1 \
    --ssh-username tdupre \
    --ssh-private-key ~/.ssh/id_rsa \
    lame10 lame11
```

**Remarks:**

- `tdupre` is my username, probably not yours.
- `localhost` can be changed to any IP address, to host the scheduler.
- `lame10 lame11` is my list of servers were I want some workers. You probably also need to change it.
- You can also give a list of server in a file: `--hostfile list_of_server.txt`, where `list_of_server.txt` contains:

```txt
lame10
lame11
```

## How to have a nice overview of your workers

You can connect to a webpage to have a nice overview of your workers:

```console
http://localhost:8787/status
```

By the way, this is why you need to install `bokeh`.

## How to use this scheduler with `joblib`

Now the cool part is that you barely have to update your joblib scripts ! Here are some examples.

**Minimal API example:**

```python
import distributed.joblib  # noqa
from joblib import parallel_backend

with parallel_backend('dask.distributed',
                      scheduler_host='localhost:8786'):
    pass  # your script using joblib
```

**Example with sklearn:**

```python
import distributed.joblib  # noqa
# scikit-learn bundles joblib, so you need to import from
# `sklearn.externals.joblib` instead of `joblib` directly
from sklearn.externals.joblib import parallel_backend
from sklearn.datasets import load_digits
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import numpy as np

digits = load_digits()

param_space = {
    'C': np.logspace(-6, 6, 13),
    'gamma': np.logspace(-8, 8, 17),
    'tol': np.logspace(-4, -1, 4),
    'class_weight': [None, 'balanced'],
}

model = SVC(kernel='rbf')
search = RandomizedSearchCV(model, param_space, cv=3, n_iter=150, verbose=10, n_jobs=-1)

with parallel_backend('dask.distributed', scheduler_host='localhost:8786'):
    search.fit(digits.data, digits.target)
```

**Remarks:**

- Be sure to check the task stream in `http://localhost:8787/status`.
- Note also that the verbose is output on the scheduler terminal, not in your script terminal.

**Other pure joblib example:**

```python
import time
import numpy as np
import distributed.joblib  # noqa
from joblib import parallel_backend, Parallel, delayed

def run(duration):
    time.sleep(duration)
    return duration

delayed_run = delayed(run)

with parallel_backend('dask.distributed',
                      scheduler_host='localhost:8786'):
    results = Parallel()(delayed_run(duration)
                         for duration in np.arange(1, 5, 0.1))
    print(np.array(results))
```

See more details in the [dask-distributed doc](https://distributed.readthedocs.io/en/latest/joblib.html)

## How to generate figures in a distant worker

As `paramiko` does not handle easily X11-forwarding (like in `shh -X`), we can't display a figure in a distant worker. However, with `matplotlib`, we can create a figure in a non-interactive backend, and save the figure with `fig.savefig('save_name.png')`.

To use a non-interactive backend, use this command _before_ importing `matplotlib.pyplot`:

```python
import matplotlib
matplotlib.use('agg')
```

Again, this command works only _before_ importing `matplotlib.pyplot`.
However, for some obscure reasons, this may fail.
You may have more luck with this command instead:

```python
import matplotlib.pyplot as plt
plt.switch_backend('agg')
```

## Advanced: How to create a different number of worker in each server

In the bash command `dask-ssh`, the number of processes (`--nprocs 1`) is identical in all servers. To have a different number of processes in each server, we need to customize `dask-ssh`.
The command `dask-ssh` is just a shortcut to a python script, [dask-ssh.py](https://github.com/dask/distributed/blob/master/distributed/cli/dask_ssh.py), so let's copy it and customize it.

For instance, let's assume we want to give the servers as a list of hostnames and integers.

```
localhost 3
lame10 2
lame11 10
```

Each line corresponds to a server and the number of processes we want in this server. We will call the script giving the list in a file: `--hostfile list_of_server.txt`.

In `dask-ssh.py`, the server list is given in the parameter `hostnames`, so we first modify the parsing to keep the lines intact:

```python
if hostfile:
    with open(hostfile) as f:
        hosts = f.readlines()
    hostnames.extend([h.split() for h in hosts])
```

Then, we give an empty list of servers to `SSHCluster`, and we start the workers manually with `start_worker`:

```python
c = SSHCluster(scheduler, scheduler_port, [], nthreads, nprocs,
               ssh_username, ssh_port, ssh_private_key, nohost,
               log_directory)

# start the workers, giving a specific number of processes if provided
for hostname in hostnames:
    if len(hostname) == 1:
        address = hostname[0]
        nprocs = c.nprocs
    else:
        address = hostname[0]
        try:
            nprocs = int(hostname[1])
        except:
            raise ValueError('Invalid hostname and number of processes %s'
                             % (hostname, ))
    c.workers.append(start_worker(c.logdir, c.scheduler_addr,
                                  c.scheduler_port, address,
                                  c.nthreads, nprocs,
                                  c.ssh_username, c.ssh_port,
                                  c.ssh_private_key, c.nohost))
```

Then we simply call the script with `python my_dask_ssh.py` instead of `dask-ssh`.

The full script is given below.


 ```python
from __future__ import print_function, division, absolute_import

from distributed.deploy.ssh import SSHCluster, start_worker
import click


@click.command(
    help="""Launch a distributed cluster over SSH. A 'dask-scheduler'
    process will run on the first host specified in [HOSTNAMES] or
    in the hostfile (unless --scheduler is specified explicitly).
    One or more 'dask-worker' processes will be run each host in
    [HOSTNAMES] or in the hostfile. Use command line flags to adjust
    how many dask-worker process are run on each host (--nprocs)
    and how many cpus are used by each dask-worker process (--nthreads).""")
@click.option('--scheduler', default=None, type=str,
              help="Specify scheduler node.  Defaults to first address.")
@click.option('--scheduler-port', default=8786, type=int,
              help="Specify scheduler port number.  Defaults to port 8786.")
@click.option('--nthreads', default=0, type=int,
              help="Number of threads per worker process. Defaults to number "
              "of cores divided by the number of processes per host.")
@click.option('--nprocs', default=1, type=int,
              help="Number of worker processes per host.  Defaults to one.")
@click.argument('hostnames', nargs=-1, type=str)
@click.option('--hostfile', default=None, type=click.Path(exists=True),
              help="Textfile with hostnames/IP addresses")
@click.option('--ssh-username', default=None, type=str,
              help="Username to use when establishing SSH connections.")
@click.option('--ssh-port', default=22, type=int,
              help="Port to use for SSH connections.")
@click.option('--ssh-private-key', default=None, type=str,
              help="Private key file to use for SSH connections.")
@click.option('--nohost', is_flag=True,
              help="Do not pass the hostname to the worker.")
@click.option('--log-directory', default=None, type=click.Path(exists=True),
              help="Directory to use on all cluster nodes for the output of "
              "dask-scheduler and dask-worker commands.")
@click.pass_context
def main(ctx, scheduler, scheduler_port, hostnames, hostfile, nthreads, nprocs,
         ssh_username, ssh_port, ssh_private_key, nohost, log_directory):
    try:
        hostnames = list(hostnames)
        if hostfile:
            with open(hostfile) as f:
                hosts = f.readlines()
            hostnames.extend([h.split() for h in hosts])

        if not scheduler:
            scheduler = hostnames[0]

    except IndexError:
        print(ctx.get_help())
        exit(1)

    c = SSHCluster(scheduler, scheduler_port, [], nthreads, nprocs,
                   ssh_username, ssh_port, ssh_private_key, nohost,
                   log_directory)

    # start the workers, giving a specific number of processes if provided
    for hostname in hostnames:
        if len(hostname) == 1:
            address = hostname[0]
            nprocs = c.nprocs
        else:
            address = hostname[0]
            try:
                nprocs = int(hostname[1])
            except:
                raise ValueError('Invalid hostname and number of processes %s'
                                 % (hostname, ))
        c.workers.append(start_worker(c.logdir, c.scheduler_addr,
                                      c.scheduler_port, address,
                                      c.nthreads, nprocs,
                                      c.ssh_username, c.ssh_port,
                                      c.ssh_private_key, c.nohost))

    import distributed
    print('\n---------------------------------------------------------------')
    print('                 Dask.distributed v{version}\n'.format(
        version=distributed.__version__))
    print('Worker nodes:'.format(n=len(hostnames)))
    for i, host in enumerate(hostnames):
        print('  {num}: {host}'.format(num=i, host=host))
    print('\nscheduler node: {addr}:{port}'.format(addr=scheduler,
                                                   port=scheduler_port))
    print(
        '---------------------------------------------------------------\n\n')

    # Monitor the output of remote processes.
    # This blocks until the user issues a KeyboardInterrupt.
    c.monitor_remote_processes()

    # Close down the remote processes and exit.
    print("\n[ dask-ssh ]: Shutting down remote processes"
          " (this may take a moment).")
    c.shutdown()
    print("[ dask-ssh ]: Remote processes have been terminated. Exiting.")


if __name__ == '__main__':
    main()
```
