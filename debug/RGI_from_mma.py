# %%
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

wolfSession = WolframLanguageSession()
# wolfSession.evaluate(wl.Get("RGI.m")) #* 1loop RGI kernel
wolfSession.evaluate(wl.Get("RGI_2loop.m")) #* 2loop RGI kernel

# wolfSession.evaluate(wl.N(1+1))

def RGInt(zeta):  
    '''
    h0 + integrate
    '''
    wolfSession.evaluate(wl.Needs("RGIPackage`"))
    res = wolfSession.evaluate(wl.RGIPackage.RGInt(zeta))
    return res


if __name__ == '__main__':
    RGInt(1.5)

