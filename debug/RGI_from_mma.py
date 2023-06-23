# %%
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

wolfSession = WolframLanguageSession()
wolfSession.evaluate(wl.Get("RGR_hard_kernel_latest.m")) 


def RGInt(zeta):  
    '''
    2 loop RGR
    '''
    wolfSession.evaluate(wl.Needs("RGRPackage`"))
    res = wolfSession.evaluate(wl.RGRPackage.RG2loop(zeta))
    return res





# wolfSession = WolframLanguageSession()
# wolfSession.evaluate(wl.Get("RGI_2loop.m")) 


# def RGInt(zeta):  
#     '''
#     2 loop RGR
#     '''
#     wolfSession.evaluate(wl.Needs("RGIPackage`"))
#     res = wolfSession.evaluate(wl.RGIPackage.RGInt(zeta))
#     return res





if __name__ == '__main__':
    RGInt(1.5)




# %%
