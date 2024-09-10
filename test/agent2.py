def agent_fn(obs, cfg):    
    # GENERATE RESPONSE
    response = "no"
    #response = np.random.choice(["yes","no"])
    if obs.keyword.lower() in obs.questions[-1].lower():
        response = "yes"

    return response
