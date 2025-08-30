from run import annahilationExtractor, evaluate_annihilation_detector

if __name__ == "__main__":
    annahilationExtractor()
    print("-"*100)
    evaluate_annihilation_detector("$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup",
                                   "Activation.sim")