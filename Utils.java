import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Thanos
 */
public class Utils {



    /**
     * Calculates transfer entropy (from y to x) assuming binary data for time
     * series X and Y
     *
     * @param X the leading time series
     * @param Y the causal time series
     * @param s the lag between the time series
     * @return
     */
    public static double transferEntropy(DoubleMatrix X, DoubleMatrix Y, int s) {

        double response = 0.0;

        int original_lentgh = X.getRows();
        int reduced_length = original_lentgh - s;

        //p(Xn+s , Xn, Yn) calculation
        DoubleMatrix TP1 = DoubleMatrix.zeros(2, 4);
        for (int i = 0; i < reduced_length; i++) {

            int Xn_s = (int) X.get(i + s, 0);
            int Xn = (int) X.get(i, 0);
            int Yn = (int) Y.get(i, 0);

            int col = Xn + 2 * Yn;
            double get = TP1.get(Xn_s, col);
            get += 1.0 / (double) reduced_length;
            TP1.put(Xn_s, col, get);
        }

        //p(Xn) calculation
        DoubleMatrix TP2 = DoubleMatrix.zeros(1, 2);
        for (int i = 0; i < original_lentgh; i++) {

            int Xn = (int) X.get(i, 0);
            double get = TP2.get(0, Xn);
            get += 1.0 / (double) original_lentgh;
            TP2.put(0, Xn, get);
        }

        //p(Xn, Yn) calculation
        DoubleMatrix TP3 = DoubleMatrix.zeros(1, 4);
        for (int i = 0; i < original_lentgh; i++) {

            int Xn = (int) X.get(i, 0);
            int Yn = (int) Y.get(i, 0);

            int col = Xn + 2 * Yn;
            double get = TP3.get(0, col);
            get += 1.0 / (double) original_lentgh;
            TP3.put(0, col, get);
        }

        //p(Xn+s, Xn) calculation
        DoubleMatrix TP4 = DoubleMatrix.zeros(1, 4);
        for (int i = 0; i < reduced_length; i++) {

            int Xn_s = (int) X.get(i + s, 0);
            int Xn = (int) X.get(i, 0);

            int col = Xn + 2 * Xn_s;
            double get = TP4.get(0, col);
            get += 1.0 / (double) reduced_length;
            TP4.put(0, col, get);
        }

        //summation               
        // i for Xn+s
        // j for Xn
        // k for Yn
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {

                    double TP1value = TP1.get(i, j + 2 * k);
                    if (TP1value == 0.0) {
                        continue;
                    }

                    double TP2value = TP2.get(0, j);
                    double TP3value = TP3.get(0, j + 2 * k);
                    double TP4value = TP4.get(0, j + 2 * i);

                    double nominator = TP1value * TP2value;
                    double denominator = TP3value * TP4value;
                    response += TP1value * Math.log10(nominator / denominator);
                }
            }
        }

        return response;
    }  
}
