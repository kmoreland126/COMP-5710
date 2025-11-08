from allpairspy import AllPairs

parameters = [
    ["file", "vuln"],                      # -a (aggregation)
    ["low", "medium", "high"],             # -l (severity level)
    ["low", "medium", "high"],             # -i (confidence level)
    ["txt", "json", "html"]                # -f (format)
]

for idx, combo in enumerate(AllPairs(parameters), 1):
    print(f"{idx}. bandit -r -a {combo[0]} --severity-level={combo[1]} --confidence-level={combo[2]} -f {combo[3]} w12/")

