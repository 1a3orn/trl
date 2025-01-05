
def select_good_responses(responses, reward_model):
    if reward_model == "fewest_es":

        def scorer(response):
            return len(response["response_str"].lower().split("e")) / len(response["response_str"])
        
        results = []
        for response in responses:
            # Response should be an array, so just sort it by 
            # the length-normalized number of "e"s.
            srt = sorted(response, key=scorer)
            #print([scorer(x) for x in srt])
            results.append(srt[0])
        return results
    else:
        raise ValueError(f"Unknown reward model: {reward_model}")