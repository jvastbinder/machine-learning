def main(infile):
    # Error file has 4 lines:
    # Actual number of true values
    # Actual number of false values
    # Number of false values predicted
    # Number of true values predicted
    errorFile = open(infile, "r")

    actualTrue = int(errorFile.readline())
    actualFalse = int(errorFile.readline())

    tp = int(errorFile.readline())
    tn = int(errorFile.readline())

    fp = actualTrue - tp
    fn = actualFalse - tn

    cPlus = tp + fn
    cMinus = tn + fn

    rPlus = tp + fp
    rMinus = tn + fn

    sensitivity = tp / cPlus
    specificity = tn / cMinus
    precisionPlus = tp / rPlus
    precisionMinus = tn / rMinus
    accuracy = (tp + tn) / (cPlus + cMinus)
    fOne = (2 * precisionPlus * sensitivity) / (sensitivity + precisionPlus)

    print("         True   class")
    print("Predicted    T    F  ")
    print("          " + str(actualTrue), str(actualFalse))
    print("True      ", tp, fp)
    print("False     ", tn, fn)
    print()
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Precision+:", precisionPlus)
    print("Precision-:", precisionMinus)
    print("Accuracy:", accuracy)
    print("F1 Score:", fOne)


main("error.txt")
