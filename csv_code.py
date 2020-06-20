import csv
def write_csv_parameters(card_X, card_Y, card_Z, mean_X, var_X, SNRdb):
    with open('E:\\6th semester\\Report\\csvfiles\\csvdist_DKL_difference_test.csv', 'w',newline='') as csvfile:
        writecsv = csv.writer(csvfile)
        writecsv.writerow(['cardX','cardY', 'cardZ', 'meanX', 'varX', 'SNR'])
        writecsv.writerow([card_X, card_Y, card_Z, mean_X, var_X, SNRdb])
        return 1