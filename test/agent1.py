def agent_fn(obs, cfg):
    
    # GENERATE RESPONSE
    keyword = "apple"
    if obs.turnType == "ask":
        response = f"Is it {keyword}?"
    else: #obs.turnType == "guess"
        response = keyword
        if obs.answers[-1] == "yes":
            response = obs.questions[-1].rsplit(" ",1)[1][:-1]

    return response
