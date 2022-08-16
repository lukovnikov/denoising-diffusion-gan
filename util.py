import argparse
import inspect
import os
import shutil
import signal
import sys
import traceback
from datetime import datetime as dt

from IPython import embed


class ticktock(object):
    """ timer-printer thingy """
    verbose_live_when_no_ttwy = False
    def __init__(self, prefix="-", verbose=True):
        self.prefix = prefix
        self.verbose = verbose
        self.state = None
        self.perc = None
        self.prevperc = None
        self._tick()
        self._tickstack = []

    def tick(self, state=None):
        self.tickpush(state=state)
        # if self.verbose and state is not None:
        #     print(f"{self.prefix}: {state}")
        # self._tick()

    def tickpush(self, state=None):
        self._tickstack.append((dt.now(), state))
        if self.verbose and state is not None:
            prefix = self._get_prefix()
            print(prefix)

    def _tick(self):
        self.ticktime = dt.now()

    def _tock(self):
        return (dt.now() - self.ticktime).total_seconds()

    def progress(self, x, of, action="", live=False):
        if self.verbose:
            self.perc = int(round(100. * x / of))
            if self.perc != self.prevperc:
                if action != "":
                    action = " " + action + " -"
                prefix = self._get_prefix()
                topr = "%s:%s %d" % (prefix, action, self.perc) + "%"
                if live:
                    self._live(topr)
                else:
                    print(topr)
                self.prevperc = self.perc

    def tock(self, action=None, prefix=None):
        self.tockpop(action=action, prefix=prefix)
        # duration = self._tock()
        # if self.verbose:
        #     prefix = prefix if prefix is not None else self.prefix
        #     action = action if action is not None else self.state
        #
        #     termsize = shutil.get_terminal_size((-1, -1))
        #     if termsize[0] > 0:
        #         left = f"{prefix}: {action}"
        #         right = f"  T: {self._getdurationstr(duration)}"
        #         print(left + right.rjust(termsize[0] - len(left) - 1), end='\n')
        #     else:
        #         print(f"{prefix}: {action} in {self._getdurationstr(duration)}")
        # return self

    def tockpop(self, action=None, prefix=None):
        if len(self._tickstack) == 0:
            print("something wrong: empty tickstack")
            return
        else:
            starttime, state = self._tickstack.pop(-1)
            duration = (dt.now() - starttime).total_seconds()
            if self.verbose:
                if prefix is None:
                    prefix = self._get_prefix()
                action = action if action is not None else state

                termsize = shutil.get_terminal_size((-1, -1))
                if termsize[0] > 0:
                    left = f"{prefix}: {action}"
                    right = f"  T: {self._getdurationstr(duration)}"
                    print(left + right.rjust(termsize[0] - len(left) - 1), end='\n')
                else:
                    print(f"{prefix}: {action} in {self._getdurationstr(duration)}")

    def msg(self, action=None, prefix=None):
        if self.verbose:
            if prefix is None:
                prefix = self._get_prefix()
            # prefix = prefix if prefix is not None else self.prefix
            action = action if action is not None else self.state
            print(f"{prefix}: {action}")
        return self

    def _get_prefix(self):
        ret = [self.prefix]
        for _, f in self._tickstack:
            if f is None:
                f = "-"
            ret.append(f)
        prefix = ": ".join(ret)
        return prefix

    def _getdurationstr(self, duration):
        if duration >= 60:
            duration = int(round(duration))
            seconds = duration % 60
            minutes = (duration // 60) % 60
            hours = (duration // 3600) % 24
            days = duration // (3600*24)
            acc = []
            if seconds > 0:
                acc.append(f"{seconds} sec")
            if minutes > 0:
                acc.append(f"{minutes} min")
            if hours > 0:
                acc.append(f"{hours} hr" + ("s" if hours > 1 else ""))
            if days > 0:
                acc.append(f"{days} day" + ("s" if days > 1 else ""))
            acc = acc[::-1]
            acc = acc[:2]
            acc = ", ".join(acc)
            return acc
        else:
            return ("%.1f sec" % duration)

    def _live(self, x, right=None):
        if right:
            termsize = shutil.get_terminal_size((-1, -1))
            if termsize[0] > 0:
                right = "  " + right
                print(f"\r{x}" + right.rjust(termsize[0] - len(x) - 1), end='')
                # sys.stdout.write(f"\r{x}" + right.rjust(termsize[0] - len(x) - 1))
            else:
                if self.verbose_live_when_no_ttwy:
                    print(f"\r{x} \t {right}", end='')
        else:
            print(f"\r{x}", end='')
            # sys.stdout.write(f"\r{x}")
        # sys.stdout.flush()

    def live(self, x):
        if self.verbose:
            prefix = self._get_prefix()
            self._live(f"{prefix}: {x}", f"T: {self._getdurationstr(self._tock())}")

    def stoplive(self):
        if self.verbose:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def argparsify(f, test=None):
    args, _, _, defaults = inspect.getargspec(f)
    assert(len(args) == len(defaults))
    parser = argparse.ArgumentParser()
    i = 0
    for arg in args:
        argtype = type(defaults[i])
        if argtype == bool:     # convert to action
            if defaults[i] == False:
                action="store_true"
            else:
                action="store_false"
            parser.add_argument("-%s" % arg, "--%s" % arg, action=action, default=defaults[i])
        else:
            parser.add_argument("-%s"%arg, "--%s"%arg, type=type(defaults[i]))
        i += 1
    if test is not None:
        par = parser.parse_args([test])
    else:
        par = parser.parse_args()
    kwargs = {}
    for arg in args:
        if getattr(par, arg) is not None:
            kwargs[arg] = getattr(par, arg)
    return kwargs


def argprun(f, sigint_shell=True, **kwargs):   # command line overrides kwargs
    """ use this to enable command-line access to kwargs of function (useful for main run methods) """
    def handler(sig, frame):
        # find the frame right under the argprun
        print("custom handler called")
        original_frame = frame
        current_frame = original_frame
        previous_frame = None
        stop = False
        while not stop and current_frame.f_back is not None:
            previous_frame = current_frame
            current_frame = current_frame.f_back
            if "_FRAME_LEVEL" in current_frame.f_locals \
                and current_frame.f_locals["_FRAME_LEVEL"] == "ARGPRUN":
                stop = True
        if stop:    # argprun frame found
            __toexposelocals = previous_frame.f_locals     # f-level frame locals
            class L(object):
                pass
            l = L()
            for k, v in __toexposelocals.items():
                setattr(l, k, v)
            stopprompt = False
            while not stopprompt:
                whattodo = input("(s)hell, (k)ill\n>>")
                if whattodo == "s":
                    embed()
                elif whattodo == "k":
                    "Killing"
                    sys.exit()
                else:
                    stopprompt = True

    if sigint_shell:
        _FRAME_LEVEL="ARGPRUN"
        prevhandler = signal.signal(signal.SIGINT, handler)
    try:
        f_args = argparsify(f)
        for k, v in kwargs.items():
            if k not in f_args:
                f_args[k] = v
        f(**f_args)

        try:
            with open(os.devnull, 'w') as f:
                oldstdout = sys.stdout
                sys.stdout = f
                from pygame import mixer
                sys.stdout = oldstdout
            mixer.init()
            mixer.music.load(os.path.join(os.path.dirname(__file__), "../resources/jubilation.mp3"))
            mixer.music.play()
        except Exception as e:
            pass
    except KeyboardInterrupt as e:
        print("Interrupted by Keyboard")
    except Exception as e:
        traceback.print_exc()
        try:
            with open(os.devnull, 'w') as f:
                oldstdout = sys.stdout
                sys.stdout = f
                from pygame import mixer
                sys.stdout = oldstdout
            mixer.init()
            mixer.music.load(os.path.join(os.path.dirname(__file__), "../resources/job-done.mp3"))
            mixer.music.play()
        except Exception as e:
            pass
# endregion

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
