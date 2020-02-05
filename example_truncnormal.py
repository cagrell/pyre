#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

# import pyre library

import time
import datetime
import sys, os
start_time = time.time()


def example_limitstatefunction(X1, X2, X3):
    """
    example limit state function
    """
    return 1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2


# Define a main() function.
def main():

    dir_pyre = 'C:\\Data\\git repos\\pyre\\'
    sys.path.insert(0, dir_pyre) # The custom directory will be used instead of original pyre
    import pyre

    # Define limit state function
    # - case 1: define directly as lambda function
    #limit_state = LimitState(lambda X1,X2,X3: 1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2)
    # - case 2: use predefined function
    limit_state = pyre.LimitState(example_limitstatefunction)

    # Set some options (optional)
    options = pyre.AnalysisOptions()
    # options.printResults(False)

    stochastic_model = pyre.StochasticModel()
    # Define random variables
    stochastic_model.addVariable(pyre.Lognormal('X1', 500, 100))
    #stochastic_model.addVariable(pyre.Normal('X2', 2000, 400))
    stochastic_model.addVariable( pyre.Truncated_Normal('X2',2000,400, 0, 10000) )
    stochastic_model.addVariable(pyre.Uniform('X3', 5, 0.5))

    # If the random variables are correlatet, then define a correlation matrix,
    # else no correlatin matrix is needed
    stochastic_model.setCorrelation(pyre.CorrelationMatrix([[1.0, 0.3, 0.2],
                                                       [0.3, 1.0, 0.2],
                                                       [0.2, 0.2, 1.0]]))

    # Performe FORM analysis
    Analysis = pyre.Form(analysis_options=options,
                    stochastic_model=stochastic_model, limit_state=limit_state)

    # # Performe Distribution analysis
    # Analysis = DistributionAnalysis(
    #     analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    # # Performe Crude Monte Carlo Simulation
    # Analysis = CrudeMonteCarlo(
    #     analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)

    # # Performe Importance Sampling
    # Analysis = ImportanceSampling(
    #     analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)

    # Some single results:
    beta = Analysis.getBeta()
    failure = Analysis.getFailure()

    print("Beta is {}, corresponding to a failure probability of {}".format(beta, failure))
    run_time = time.time() - start_time
    print(str(datetime.timedelta(seconds=run_time)))

    # This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
