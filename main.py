import csv 
import numpy
import scipy.stats

# Input parameters
STRESS_DATA    : str         = "stress"
FILENAME       : list[str]   = ["FTSE100","SSE","BP","RIO","HSBA","BARC", "LIN"]
WEIGHT         : list[float] = [-0.0, .23, .23, .23, .11, .10, .10]  # NO SWAP
WEIGHT         : list[float] = [-0.31, .16, .16, .16, .07, .07, .07] # SWAP COMMITTED
#WEIGHT = [-.13, .19, .19, .19, .1,.1,.1]

# Simulation parameters.
IS_STRESS          : bool        = True
NUMBER_PERIODS     : int         = 1
TOTAL_SIMULATIONS  : int         = 1000
TIME_SIMULATED     : int         = 30
TIME_FRAMES        : list[int]   = [5, 15, 20, 28] 
RETURN_PERIODS     : list[int]   = [192, 96, 48, 24, 12, 6]

# Global variables 
PATH               : str                = "/home/bernardo/Desktop/stockVaR/stocks/"
CSV                : str                = ".csv"
SUFFIX             : str                = ""
TOTAL_COUNT        : int                = len (FILENAME)

# Variable declaraion
data               : list[list[float]]       = [[] for _ in range(TOTAL_COUNT)]
stress_data        : list[float]             = []
mean               : list[float]             = [None for _ in range(TOTAL_COUNT)]
standard_deviation : list[float]             = [None for _ in range(TOTAL_COUNT)]
covariance_matrix  : list[list[float]]       = [[None for _ in range(TOTAL_COUNT)] for _ in range(TOTAL_COUNT)]
copula_mean        : list[float]             = [0 for _ in range (TOTAL_COUNT)]
unprocessed        : list [dict [str,float]] = [dict () for _ in range(TOTAL_COUNT)]
date_stamp         : list [str]              = []
portfolio_value    : list[list[float]]       = [[0 for _ in range (TIME_SIMULATED)] for _ in range (TOTAL_SIMULATIONS)]

# Extraction of data from file.
for i in range(TOTAL_COUNT):
    with open(PATH + FILENAME[i] + SUFFIX + CSV, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\"', quotechar='|')
        # First row is not a numerical value. Skip.
        reader : list[list[float]] = list(reader)[1:]
        # For the first iteration, we need the dates.
        if i==0:
            date_stamp = [row[1] for row in reader]
        #Actually parses the rows
        for row in reader:
            try:
                # Parses to float and adds to dictionary for the corresponding 
                # variable. Note that we are expecting some data as, e.g. "1.37%".
                unprocessed[i].update ({row[1] : float (
                        # Column No.13 corresponds to the monthly % change, on file.
                        row[13].
                        # Removes characters that should not be part of the parse.
                        replace("\"","").replace("%",""))
                        # Turns to percentage.
                        /100.})
                # Common sense check. If we catch values too high, might as well check 
                # that the input data makes sense to begin with.
                value = unprocessed[i].get (row[1])
                if (value > 1):
                    print (row)
            # If parsing of any row fails, displays the row. This is likely a problem with 
            # the input data itself, that requires manual fix. However, we do not need to 
            # stop execution, as the script will simply ignore this row.
            except ValueError:
                buffer = []
                buffer.append ("Could not parse the following row: ")
                buffer.append (FILENAME[i])
                buffer.append (" : ")
                buffer.append ("".join (row))
                print ("".join (buffer))

# Now we need to check which date stamps we actually include to the data. Say file 1 and 2
# have the date "4 June 2023", but file 3 doesn't: in this case, we cannot include the 
# data concerning "4 June 2023" to our analysis, because the data-set is incomplete!
inclusion : list[bool] = [True for _ in date_stamp]
for d, date in enumerate(date_stamp):
    for i in range (TOTAL_COUNT):
        if unprocessed[i].get (date) == None:
            inclusion[d] = False
            break
# Actually includes the data the passed validation.
for date, include in zip (date_stamp, inclusion):
    if (include):
        for i in range (TOTAL_COUNT):
            data[i].append (unprocessed[i].get(date))

# If we are running a stress test, extracts data from file for stress test.
if (IS_STRESS):
    with open (PATH + STRESS_DATA + CSV, newline='') as csvfile:
        reader = csv.reader (csvfile, delimiter='\"', quotechar='|')
        for row in reader:
            stress_data.append (
                # Same processing as before.
                float (row[13].replace ("\"", ""). replace ("%", ""))/100.)
# We actually must ensure that the data is in ascending order of dates...
stress_data = stress_data[::-1]

# Calculates mean and standard deviation for each data field.
for i in range (TOTAL_COUNT):
    standard_deviation[i] = numpy.std (data[i])
    mean[i] = (numpy.mean (data[i])
        # Adjustment for GBM to be risk-neutral
        - standard_deviation[i] * standard_deviation[i] / 2.)

# Calculation of correlation matrix, i.e., (c)ij = cov (Xi,Yi) / (std(Xi) * std(Yi))
for i in range (TOTAL_COUNT):
    for j in range (TOTAL_COUNT):
        covariance_matrix[i][j] = numpy.corrcoef (
            data[i], data[j]) [0] [1]
        # This portion is meant for validation of the correlation. Comment-out for a 
        # regular simulation.
        #if (i == j):
        #    covariance_matrix[i][j] = 1
        #else:
        #    covariance_matrix[i][j] = 0

# Cholesky decomposition (lower-triangular matrix) for correlation matrix.
cholesky : numpy.ndarray[float, float] = numpy.linalg.cholesky (covariance_matrix)

# Creates the normal distributions for each asset.
distribution = [None for _ in range (TOTAL_COUNT)]
for i in range (TOTAL_COUNT):
    distribution[i] = scipy.stats.norm (
        loc=mean[i] / NUMBER_PERIODS, scale=standard_deviation[i] 
        / numpy.sqrt (NUMBER_PERIODS))

# Sets up a standard normal distribution for the sampling phase.
standard_normal = scipy.stats.norm (loc=0, scale=1)

# Monte Carlo Sampling.
for s in range (TOTAL_SIMULATIONS):

    # Creates an array containing the (accumulated) variations for each asset and time.
    variation : list [list[float]] = [[None for _ in range (TIME_SIMULATED)] 
        for _ in range(TOTAL_COUNT)]
    # Start at 100%. 
    #variation [0][0] = 0
    variation [0][0] = 1
    for i in range (1, TOTAL_COUNT):
        variation[i][0] = 1

    # Samples the accumulated variations for p time-periods.
    for p in range (1, TIME_SIMULATED):
        # Samples N quantiles from a standard normal
        quantile = standard_normal.rvs (TOTAL_COUNT)

        # If it is a stress scenario, before we correlate the quantiles, we must first
        # adjust the first quantile to match the loss corresponding to the guide index.
        if (IS_STRESS):
            # This step goes from quantile --> cumulant --> standard_normal quantile. 
            quantile_for_index = stress_data [p-1]
            accumulation = distribution[0].cdf (quantile_for_index)
            # The first quantile gets replaced by the quantile from the stress data.
            quantile[0] = standard_normal.ppf (accumulation)
            # Adjust the extreme cases slightly to prevent numeric overflow.
            if (accumulation >= 1 or accumulation == numpy.inf):
                quantile [0] =.999
            elif (accumulation <= 0 or accumulation == numpy.nan):
                quantile [0] = .001

        # Correlates & cumulates.
        correlated = numpy.matmul (cholesky, quantile)
        cumulant = standard_normal.cdf (correlated)

        # Calculates the variation for each iid, accumulated.
        #variation[0][p] = distribution[0].ppf (cumulant[0])
        for i in range (TOTAL_COUNT):
            variation[i][p] = (1 + distribution[i].ppf (
                cumulant[i]
            )) * variation [i][p-1]

    # Calculates the adjustment value for display purposes.
    adjustment = 0
    #swap = 0
    #guiding_weight = WEIGHT[0]
    #for i in range (1, TOTAL_COUNT):
    for i in range (TOTAL_COUNT):
        adjustment += variation [i][0] * WEIGHT [i]
    # Actually updates the total portfolio VaR as funciton of the return periods.
    for p in range (TIME_SIMULATED):
        values = [variation [i][p] * WEIGHT [i] for i in range (TOTAL_COUNT)]
        portfolio_value[s][p] = sum (values) - adjustment
        #portfolio_value[s][p] = sum (values [1:]) - adjustment
        # Finally, adjust the portfolio by the swap on the guiding index...
        #swap += variation[0][p] * guiding_weight
        #portfolio_value[s][p] += swap
        # For the guiding index, we actually do the update differently... This is because
        # the weight of the guiding index must be readjusted by iteration...
        #guiding_weight = WEIGHT[0] / sum (WEIGHT[1:]) * sum (values [1:])

# Gets the cdf for the portfolio value as a function of time, by transposing the original
# array and sorting the simulation VaR(s) in order.
cdf = [[None for _ in range (TOTAL_SIMULATIONS)] 
    for _ in range (TIME_SIMULATED)]
for s in range (TOTAL_SIMULATIONS):
    for p in range (TIME_SIMULATED):
        cdf[p][s] = portfolio_value [s][p]
for p in range (TIME_SIMULATED):
    cdf[p].sort ()

# Sets up the buffer(s) for the output. The output will be a CSV string.
buffer = []
sub_buffer = []
# Table headers
buffer.append ("Return Period")
buffer.append (",")
buffer.append ("Value-at-risk")
buffer.append ("\n")
buffer.append (",")
# Time-frames of interest - sub-header row.
for t in TIME_FRAMES:
    buffer.append ("Day: {t},".format(t=t))
# Mean row.
buffer.append ("\n")
buffer.append ("mean")
buffer.append (",")
for t in TIME_FRAMES:
    mean = round (numpy.mean (cdf [t]),4)
    buffer.append (str (mean))
    buffer.append (",")
# Stddev row.
buffer.append ("\n")
buffer.append ("stddev")
buffer.append (",")
for t in TIME_FRAMES:
    stddev = round (numpy.std (cdf [t]), 4)
    buffer.append (str (stddev))
    buffer.append (",")
# The actual return periods. 
for rp in RETURN_PERIODS:
    sub_sub_buffer = []
    buffer.append ("\n")
    # Label column.
    buffer.append ("1-in-{rp}".format (rp=rp))
    buffer.append (",")
    # The sub_buffer will store the disfavourable cases, so we can append it to the main
    # buffer in the end.
    sub_sub_buffer.append ("\n")
    sub_sub_buffer.append ("1-in-{rp}".format (rp=rp))
    sub_sub_buffer.append (",")
    for t in TIME_FRAMES:
        # Gets the VaR metrics for favourable and disfavourable cases required.
        disfavourable = round (cdf [t][int(1/rp*TOTAL_SIMULATIONS)],4)
        favourable = round (cdf [t][int((1-1/rp)*TOTAL_SIMULATIONS)],4) 
        buffer.append (str (favourable))
        buffer.append (",")
        sub_sub_buffer.append (str (disfavourable))
        sub_sub_buffer.append (",")
    sub_buffer.append ("".join (sub_sub_buffer))
# Add the disfavourable cases to the buffer.
buffer.extend (sub_buffer [::-1])
# Streams buffer to string.
buffer = "".join (buffer)
# Prints output.
print (buffer)

# # Displays the covariance matrix.
# buffer = []
# buffer.append (",".join (FILENAME))
# buffer.append ("\n")
# for row in covariance_matrix:
#     for element in row:
#         buffer.append (str(round (element, 4)))
#         buffer.append (",")
#     buffer.append ("\n")
# print ("".join(buffer))
# # # The determinant.
# # print (numpy.linalg.det(covariance_matrix))
