import logging

logging.getLogger("rpy2").setLevel(logging.ERROR)

import rpy2
import rpy2.robjects as ro

from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# Disable rpy2 warnings
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda *args: None
rpy2.rinterface_lib.init_ = ["--quiet", "--no-save"]


def dnadist(fasta_path, model, gamma=False):
    with localconverter(ro.default_converter + pandas2ri.converter):
        importr("ape")

        ro.globalenv["fasta_path"] = fasta_path
        ro.globalenv["model"] = model
        ro.globalenv["gamma"] = gamma

        dm = ro.r(
            """
            aln <- read.FASTA(fasta_path, type = "DNA")

            dm <- dist.dna(aln, model = model, gamma = gamma)

            D <- as.data.frame(as.matrix(dm))
            D
            """
        )

    return dm
