"""
Minimum working example of BEAGLE4State
to calculate the likelihood of a Phylo2Vec matrix.

This likelihood function could be used in MCMC
to optimize a tree topology or branch lengths.

A simple MCMC scheme could be to sample a matrix,
make a topology and branch length change on an index of the matrix,
and accept the change with a probability based on the difference
in tree likelihoods.

Based on this example from the BEAGLE library:
https://github.com/beagle-dev/beagle-lib/blob/hmc-clock/examples/standalone/hellobeagle/hello.cpp

License: MIT License

Adaptation in Python made by Neil Scheidwasser, 2025
"""

# pylint: disable=redefined-outer-name, invalid-name
import math

import numba as nb
import numpy as np

from numba import int64, float64
from numba.experimental import jitclass

from phylo2vec.base.ancestry import to_ancestry

BEAGLE_OP_NONE = -1
P_PAD = 0
T_PAD = 1
OFFSET = 4 + T_PAD


@nb.njit(cache=True)
def createStates(data):
    TABLE = {v: k for k, v in enumerate("acgt-")}

    # Temporary (ugly) fixes for non-standard characters
    # R is a purine, so we map it to A
    TABLE["r"] = TABLE["a"]
    # Y is a pyrimidine, so we map it to C
    TABLE["y"] = TABLE["c"]
    # M is an amino, so we map it to A
    TABLE["m"] = TABLE["a"]
    # K is a keto, so we map it to G
    TABLE["k"] = TABLE["g"]

    # Replacing non-standard characters with '-' is not great
    # but temporary fix until we have a better way to handle them
    return [TABLE[x] for x in data.lower()]


def vToOperations(v):
    ancestry = to_ancestry(v)
    operations = np.empty((ancestry.shape[0], 7), dtype=np.int64)

    operations.fill(BEAGLE_OP_NONE)

    operations[:, 0] = ancestry[:, -1]
    operations[:, 3] = operations[:, 4] = ancestry[:, 0]
    operations[:, 5] = operations[:, 6] = ancestry[:, 1]

    return operations


@nb.njit(cache=True)
def PREFETCH_MATRIX(matrices, w):
    # Extract the 4x4 matrix values from the flat list
    m00 = matrices[w + OFFSET * 0 + 0]
    m01 = matrices[w + OFFSET * 0 + 1]
    m02 = matrices[w + OFFSET * 0 + 2]
    m03 = matrices[w + OFFSET * 0 + 3]

    m10 = matrices[w + OFFSET * 1 + 0]
    m11 = matrices[w + OFFSET * 1 + 1]
    m12 = matrices[w + OFFSET * 1 + 2]
    m13 = matrices[w + OFFSET * 1 + 3]

    m20 = matrices[w + OFFSET * 2 + 0]
    m21 = matrices[w + OFFSET * 2 + 1]
    m22 = matrices[w + OFFSET * 2 + 2]
    m23 = matrices[w + OFFSET * 2 + 3]

    m30 = matrices[w + OFFSET * 3 + 0]
    m31 = matrices[w + OFFSET * 3 + 1]
    m32 = matrices[w + OFFSET * 3 + 2]
    m33 = matrices[w + OFFSET * 3 + 3]

    # Return the values as a 4x4 nested list
    return np.array(
        [
            [m00, m01, m02, m03],
            [m10, m11, m12, m13],
            [m20, m21, m22, m23],
            [m30, m31, m32, m33],
        ]
    )


@nb.njit(cache=True)
def PREFETCH_PARTIALS(partials, v):
    return partials[v : v + 4]


@nb.njit(cache=True)
def DO_INTEGRATION(m, p):
    # https://github.com/numba/numba/issues/8739#issuecomment-1419010762
    return m @ p[::1]


eigenDecompositionSpec = [
    ("kDecompositionCount", int64),
    ("kStateCount", int64),
    ("kCategoryCount", int64),
    ("gEigenValues", float64[:, :]),
    ("gCMatrices", float64[:, :]),
    ("matrixTmp", float64[:]),
]


@jitclass(eigenDecompositionSpec)
class EigenDecompositionCube:
    def __init__(self, decompositionCount, statecount, categoryCount):
        self.kDecompositionCount = decompositionCount
        self.kStateCount = statecount
        self.kCategoryCount = categoryCount

        self.gEigenValues = np.empty((self.kDecompositionCount, self.kStateCount))
        self.gCMatrices = np.empty((self.kDecompositionCount, self.kStateCount**3))

        self.matrixTmp = np.empty((self.kStateCount,))

    def setEigenDecomposition(self, eigenIndex, model):
        """Set the eigendecomposition for a given matrix

        Parameters
        ----------
        eigenIndex : int
            Index of eigendecomposition buffer
        model : str
            DNA Substitution model
        """
        inEigval, inEigvec, inInvEigvec = self.eig(model)
        ll = 0
        for i in range(self.kStateCount):
            self.gEigenValues[eigenIndex][i] = inEigval[i]
            for j in range(self.kStateCount):
                for k in range(self.kStateCount):
                    self.gCMatrices[eigenIndex][ll] = inEigvec[i, k] * inInvEigvec[k, j]
                    ll += 1

    @staticmethod
    def eig(model, x=1 / 3):
        """Get eigenvalues and eigenvectors for a given model

        Parameters
        ----------
        model : str
            DNA Substitution model
        x : float, optional
            Substitution rate, by default 1/3

            For now, only JC69 model is implemented, so only one parameter is necessary

        Returns
        -------
        eigval: np.ndarray
            Eigenvalues
        eigvec: np.ndarray
            Eigenvectors
        inveigvec: np.ndarray
            Inverse of eigenvectors
        """
        if model == "JC69":
            Q = x * np.ones((4, 4)) - 4 * x * np.eye(4)

            eigval, eigvec = np.linalg.eig(Q)

            inveigvec = np.linalg.inv(eigvec)
        else:
            raise NotImplementedError("Model not implemented")

        return eigval, eigvec, inveigvec

    def updateTransitionMatrices(
        self,
        eigenIndex,
        probabilityIndices,
        edgeLengths,
        categoryRates,
        transitionMatrices,
        count,
    ):
        """
        Calculate a transition probability matrices for a given list of nodes.
        This will calculate for all categories (and all matrices if more than one is being used).

        Parameters
        ----------
        eigenIndex : int
            Index of eigendecomposition buffer
        probabilityIndices : np.ndarray
            Node indices
        edgeLengths : np.ndarray
            Branch lengths
        categoryRates : np.ndarray
            Rate scalers
        transitionMatrices : np.ndarray
            Transition matrices
        count : int
            The number of nodes
        """
        stateCountModFour = (self.kStateCount // 4) * 4

        for u in range(count):
            transitionMat = transitionMatrices[probabilityIndices[u]]
            n = 0
            for ll in range(self.kCategoryCount):
                for i in range(self.kStateCount):
                    self.matrixTmp[i] = math.exp(
                        self.gEigenValues[eigenIndex][i]
                        * edgeLengths[u]
                        * categoryRates[ll]
                    )

                tmpCMatrices = self.gCMatrices[eigenIndex]

                for i in range(self.kStateCount):
                    for _ in range(self.kStateCount):
                        sum_ = 0.0

                        k = 0
                        while k < stateCountModFour:
                            sum_ += tmpCMatrices[k] * self.matrixTmp[k]
                            sum_ += tmpCMatrices[k + 1] * self.matrixTmp[k + 1]
                            sum_ += tmpCMatrices[k + 2] * self.matrixTmp[k + 2]
                            sum_ += tmpCMatrices[k + 3] * self.matrixTmp[k + 3]
                            k += 4

                        while k < self.kStateCount:
                            sum_ += tmpCMatrices[k] * self.matrixTmp[k]
                            k += 1
                        tmpCMatrices = tmpCMatrices[self.kStateCount :]

                        if sum_ > 0:
                            transitionMat[n] = sum_
                        else:
                            transitionMat[n] = 0
                        n += 1
                    if T_PAD != 0:
                        transitionMat[n] = 1.0
                        n += T_PAD


# pylint: disable=no-member
beagle4StateSpec = [
    ("kTipCount", int64),
    ("kBufferCount", int64),
    ("kPatternCount", int64),
    ("kStateCount", int64),
    ("kPaddedPatternCount", int64),
    ("kExtraPatterns", int64),
    ("kEigenDecompCount", int64),
    ("kMatrixCount", int64),
    ("kMatrixSize", int64),
    ("kCategoryCount", int64),
    ("kPaddedPatternCount", int64),
    ("kPartialsPaddedStateCount", int64),
    ("kPartialsSize", int64),
    ("gTipStates", int64[:, :]),
    ("gPatternWeights", float64[:]),
    ("gStateFrequencies", float64[:, :]),
    ("gCategoryWeights", float64[:, :]),
    ("gCategoryRates", float64[:, :]),
    ("kScaleBufferCount", int64),
    ("gScaleBuffers", float64[:, :]),
    ("gEigenDecompositions", EigenDecompositionCube.class_type.instance_type),
    ("gTransitionMatrices", float64[:, :]),
    ("gPartials", float64[:, :]),
    ("integrationTmp", float64[:]),
    ("outLogLikelihoodsTmp", float64[:]),
]
# pylint: enable=no-member


@jitclass(beagle4StateSpec)
class Beagle4State:
    def __init__(
        self,
        n_tips,
        n_internal,
        n_buffers,
        n_states,
        n_patterns,
        n_eigendecomposition_buffers,
        n_transition_buffers,
        n_rate_cats,
        n_scaling_buffers,
        # device,
        # n_devices,
    ):
        self.kTipCount = n_tips
        self.kBufferCount = n_internal + n_buffers
        self.kPatternCount = n_patterns
        self.kStateCount = n_states
        self.kPaddedPatternCount = self.kPatternCount
        # int modulus = getPaddedPatternsModulus();
        # int remainder = kPatternCount % modulus;
        # if (remainder != 0) {
        #     kPaddedPatternCount += modulus - remainder;
        # }
        self.kExtraPatterns = self.kPaddedPatternCount - self.kPaddedPatternCount
        self.kEigenDecompCount = n_eigendecomposition_buffers
        self.kMatrixCount = n_transition_buffers
        self.kMatrixSize = (T_PAD + self.kStateCount) * self.kStateCount
        self.kCategoryCount = n_rate_cats
        self.kPaddedPatternCount = self.kPatternCount
        self.kPartialsPaddedStateCount = self.kStateCount + P_PAD
        self.kPartialsSize = (
            self.kPaddedPatternCount
            * self.kPartialsPaddedStateCount
            * self.kCategoryCount
        )

        self.gTipStates = np.full(
            (self.kBufferCount, self.kPaddedPatternCount),
            BEAGLE_OP_NONE,
            dtype=np.int64,
        )
        self.gPatternWeights = np.empty((self.kPatternCount,))
        self.gStateFrequencies = np.empty((self.kEigenDecompCount, self.kStateCount))
        self.gCategoryWeights = np.empty((self.kEigenDecompCount, 1))
        self.gCategoryRates = np.empty((self.kEigenDecompCount, 1))
        scaleBufferSize = self.kPaddedPatternCount
        self.kScaleBufferCount = n_scaling_buffers
        self.gScaleBuffers = np.empty((self.kScaleBufferCount, scaleBufferSize))

        self.gEigenDecompositions = EigenDecompositionCube(
            self.kEigenDecompCount, self.kStateCount, self.kCategoryCount
        )
        self.gTransitionMatrices = np.empty(
            (self.kMatrixCount, self.kMatrixSize * self.kCategoryCount)
        )

        self.gPartials = np.empty((self.kBufferCount, self.kPartialsSize))

        self.integrationTmp = np.zeros((self.kPatternCount * self.kStateCount))

        self.outLogLikelihoodsTmp = np.zeros((self.kPatternCount * self.kStateCount))

    def setTipStates(self, tipIndex, inStates):
        """Set the states for a given tip

        Parameters
        ----------
        tipIndex : int
            index of the tip
        inStates : np.ndarray
            array of states. Range = 0 to stateCount - 1, missing = stateCount
        """
        if tipIndex < 0 or tipIndex >= self.kTipCount:
            raise ValueError("Invalid tip index")
        for j in range(self.kPatternCount):
            self.gTipStates[tipIndex][j] = min(inStates[j], self.kStateCount)
        for j in range(self.kPatternCount, self.kPaddedPatternCount):
            self.gTipStates[tipIndex][j] = self.kStateCount

    def setPatternWeights(self, inPatternWeights):
        """Set pattern weights

        Parameters
        ----------
        inPatternWeights : np.ndarray of size (self.kPatternCount,)
            pattern weights
        """
        self.gPatternWeights = inPatternWeights

    def setStateFrequencies(self, stateFrequenciesIndex, inFreqs):
        if stateFrequenciesIndex < 0 or stateFrequenciesIndex >= self.kEigenDecompCount:
            raise ValueError("Invalid state frequencies index")
        self.gStateFrequencies[stateFrequenciesIndex] = inFreqs

    def setCategoryWeights(self, categoryWeightsIndex, weights):
        self.gCategoryWeights[categoryWeightsIndex] = weights

    def setCategoryRates(self, inCategoryRates):
        """Set category rates

        Parameters
        ----------
        inCategoryRates : np.ndarray
            Rate scalers
        """
        self.gCategoryRates[0] = inCategoryRates

    def setEigenDecomposition(self, eigenIndex, model):
        """see EigenDecompositionCube.setEigenDecomposition"""
        self.gEigenDecompositions.setEigenDecomposition(eigenIndex, model)

    def updateTransitionMatrices(
        self, eigenIndex, probabilityIndices, edgeLengths, count
    ):
        """see EigenDecompositionCube.updateTransitionMatrices"""
        self.gEigenDecompositions.updateTransitionMatrices(
            eigenIndex,
            probabilityIndices,
            edgeLengths,
            self.gCategoryRates[0],
            self.gTransitionMatrices,
            count,
        )

    def updatePartials(self, operations, operationCount, cumulativeScaleIndex):
        """Calculate or queue for calculation partials using an array of operations

        Parameters
        ----------
        operations : np.ndarray
            An array of triplets of indices = two children and a parent
        operationCount : int
            Number of operations
        cumulativeScaleIndex : int
            Indicates if partials should be rescaled
        """
        cumulativeScaleBuffer = None
        if cumulativeScaleIndex != BEAGLE_OP_NONE:
            cumulativeScaleBuffer = self.gScaleBuffers[cumulativeScaleIndex]

        for op in range(operationCount):
            parIndex = operations[op][0]
            writeScalingIndex = operations[op][1]
            readScalingIndex = operations[op][2]
            child1Index = operations[op][3]
            child1TransMatIndex = operations[op][4]
            child2Index = operations[op][5]
            child2TransMatIndex = operations[op][6]

            currentPartition = 0

            partials1 = self.gPartials[child1Index]
            partials2 = self.gPartials[child2Index]

            tipStates1 = self.gTipStates[child1Index]
            tipStates2 = self.gTipStates[child2Index]

            matrices1 = self.gTransitionMatrices[child1TransMatIndex]
            matrices2 = self.gTransitionMatrices[child2TransMatIndex]

            destPartials = self.gPartials[parIndex]

            startPattern = 0
            endPattern = self.kPatternCount

            rescale = BEAGLE_OP_NONE

            if np.all(tipStates1 != BEAGLE_OP_NONE):
                if np.all(tipStates2 != BEAGLE_OP_NONE):
                    if rescale == 0:
                        raise NotImplementedError("calcStatesStatesFixedScaling")
                    else:
                        self.calcStatesStates(
                            destPartials,
                            tipStates1,
                            matrices1,
                            tipStates2,
                            matrices2,
                            startPattern,
                            endPattern,
                        )
                        if rescale == 1:
                            raise NotImplementedError("rescalePartials")
                else:
                    if rescale == 0:
                        raise NotImplementedError("calcStatesPartialsFixedScaling")
                    else:
                        self.calcStatesPartials(
                            destPartials,
                            tipStates1,
                            matrices1,
                            partials2,
                            matrices2,
                            startPattern,
                            endPattern,
                        )
                        if rescale == 1:
                            raise NotImplementedError("rescalePartials")
            else:
                if np.all(tipStates2 != BEAGLE_OP_NONE):
                    if rescale == 0:
                        raise NotImplementedError("calcStatesPartialsFixedScaling")
                    else:
                        self.calcStatesPartials(
                            destPartials,
                            tipStates2,
                            matrices2,
                            partials1,
                            matrices1,
                            startPattern,
                            endPattern,
                        )
                        if rescale == 1:
                            raise NotImplementedError("rescalePartials")
                else:
                    if rescale == 2:
                        raise NotImplementedError("calcPartialsPartialsAutoScaling")
                    elif rescale == 0:
                        raise NotImplementedError("calcPartialsPartialsFixedScaling")
                    else:
                        self.calcPartialsPartials(
                            destPartials,
                            partials1,
                            matrices1,
                            partials2,
                            matrices2,
                            startPattern,
                            endPattern,
                        )
                        if rescale == 1:
                            raise NotImplementedError("rescalePartials")

    def calcStatesStates(
        self, destP, states1, matrices1, states2, matrices2, startPattern, endPattern
    ):
        """Calculates partial likelihoods at a node when both children have states

        Parameters
        ----------
        destP : np.ndarray
            Array to store output partials
        states1 : np.ndarray
            States for child 1
        matrices1 : np.ndarray
            Transition matrix for child 1
        states2 : np.ndarray
            States for child 2
        matrices2 : np.ndarray
            Transition matrix for child 2
        startPattern : int
            Index of the first pattern
        endPattern : int
            Index of the last pattern
        """
        for ll in range(self.kCategoryCount):
            v = ll * 4 * self.kPaddedPatternCount

            if startPattern != 0:
                v += 4 * startPattern

            w = ll * 4 * OFFSET
            for k in range(startPattern, endPattern):
                state1 = states1[k]
                state2 = states2[k]

                destP[v] = matrices1[w + state1] * matrices2[w + state2]
                destP[v + 1] = (
                    matrices1[w + OFFSET + state1] * matrices2[w + OFFSET + state2]
                )
                destP[v + 2] = (
                    matrices1[w + OFFSET * 2 + state1]
                    * matrices2[w + OFFSET * 2 + state2]
                )
                destP[v + 3] = (
                    matrices1[w + OFFSET * 3 + state1]
                    * matrices2[w + OFFSET * 3 + state2]
                )

                v += 4

    def calcStatesPartials(
        self, destP, states1, matrices1, partials2, matrices2, startPattern, endPattern
    ):
        """Calculates partial likelihoods at a node when one child has states and one has partials

        Parameters
        ----------
        destP : np.ndarray
            Array to store output partials
        states1 : np.ndarray
            States for child 1
        matrices1 : np.ndarray
            Transition matrix for child 1
        partials2 : np.ndarray
            Partials for child 2
        matrices2 : np.ndarray
            Transition matrix for child 2
        startPattern : int
            Index of the first pattern
        endPattern : int
            Index of the last pattern
        """
        for ll in range(self.kCategoryCount):
            u = ll * 4 * self.kPaddedPatternCount

            if startPattern != 0:
                u += 4 * startPattern

            w = ll * 4 * OFFSET

            m = PREFETCH_MATRIX(matrices2, w)

            for k in range(startPattern, endPattern):
                state1 = states1[k]
                p = PREFETCH_PARTIALS(partials2, u)

                sums = DO_INTEGRATION(m, p)

                destP[u] = matrices1[w + state1] * sums[0]
                destP[u + 1] = matrices1[w + OFFSET + state1] * sums[1]
                destP[u + 2] = matrices1[w + OFFSET * 2 + state1] * sums[2]
                destP[u + 3] = matrices1[w + OFFSET * 3 + state1] * sums[3]

                u += 4

    def calcPartialsPartials(
        self,
        destP,
        partials1,
        matrices1,
        partials2,
        matrices2,
        startPattern,
        endPattern,
    ):
        """Calculates partial likelihoods at a node when both children have partials

        Parameters
        ----------
        destP : np.ndarray
            Array to store output partials
        partials1 : np.ndarray
            Partials for child 1
        matrices1 : np.ndarray
            Transition matrix for child 1
        partials2 : np.ndarray
            Partials for child 2
        matrices1 : np.ndarray
            Transition matrix for child 2
        startPattern : int
            Index of the first pattern
        endPattern : int
            Index of the last pattern
        """
        for ll in range(self.kCategoryCount):
            u = ll * 4 * self.kPaddedPatternCount
            if startPattern != 0:
                u += 4 * startPattern
            w = ll * 4 * OFFSET

            m1 = PREFETCH_MATRIX(matrices1, w)
            m2 = PREFETCH_MATRIX(matrices2, w)

            for _ in range(startPattern, endPattern):
                p1 = PREFETCH_PARTIALS(partials1, u)
                p2 = PREFETCH_PARTIALS(partials2, u)

                sum10, sum11, sum12, sum13 = DO_INTEGRATION(m1, p1)
                sum20, sum21, sum22, sum23 = DO_INTEGRATION(m2, p2)

                destP[u] = sum10 * sum20
                destP[u + 1] = sum11 * sum21
                destP[u + 2] = sum12 * sum22
                destP[u + 3] = sum13 * sum23

                u += 4

    def calcRootLogLikelihoods(
        self,
        bufferIndex,
        categoryWeightsIndex,
        stateFrequenciesIndex,
        scalingFactorsIndex,
    ):
        rootPartials = self.gPartials[bufferIndex]

        wt = self.gCategoryWeights[categoryWeightsIndex]

        u = 0
        v = 0
        wt0 = wt[0]

        for _ in range(self.kPatternCount):
            self.integrationTmp[v] += rootPartials[v] * wt0
            self.integrationTmp[v + 1] += rootPartials[v + 1] * wt0
            self.integrationTmp[v + 2] += rootPartials[v + 2] * wt0
            self.integrationTmp[v + 3] += rootPartials[v + 3] * wt0
            v += 4

        for ll in range(1, self.kCategoryCount):
            u = 0
            wtl = wt[ll]
            for _ in range(self.kPatternCount):
                self.integrationTmp[u] += rootPartials[v] * wtl
                self.integrationTmp[u + 1] += rootPartials[v + 1] * wtl
                self.integrationTmp[u + 2] += rootPartials[v + 2] * wtl
                self.integrationTmp[u + 3] += rootPartials[v + 3] * wtl

                u += 4
                v += 4

            v += 4 * self.kExtraPatterns

        return self.integrateOutStatesAndScale(
            stateFrequenciesIndex, scalingFactorsIndex
        )

    def integrateOutStatesAndScale(self, stateFrequenciesIndex, scalingFactorsIndex):
        freq0 = self.gStateFrequencies[stateFrequenciesIndex][0]
        freq1 = self.gStateFrequencies[stateFrequenciesIndex][1]
        freq2 = self.gStateFrequencies[stateFrequenciesIndex][2]
        freq3 = self.gStateFrequencies[stateFrequenciesIndex][3]

        u = 0

        for k in range(self.kPatternCount):
            sumOverI = (
                freq0 * self.integrationTmp[u]
                + freq1 * self.integrationTmp[u + 1]
                + freq2 * self.integrationTmp[u + 2]
                + freq3 * self.integrationTmp[u + 3]
            )

            u += 4

            self.outLogLikelihoodsTmp[k] = math.log(sumOverI)

        if scalingFactorsIndex != BEAGLE_OP_NONE:
            scalingFactors = self.gScaleBuffers[scalingFactorsIndex]
            for k in range(self.kPatternCount):
                self.outLogLikelihoodsTmp[k] += scalingFactors[k]

        outSumLogLikelihood = 0
        for k in range(self.kPatternCount):
            outSumLogLikelihood += (
                self.outLogLikelihoodsTmp[k] * self.gPatternWeights[k]
            )

        return outSumLogLikelihood


def parse_matrix(m):
    v = m[:, 0].astype(int)
    edge_lengths = m[:, 1:]

    edge_lengths_flattened = np.zeros((2 * len(v),))
    ancestry = to_ancestry(v)

    for i in range(ancestry.shape[0] - 1, -1, -1):
        edge_lengths_flattened[ancestry[i, 0]] = edge_lengths[i, 0]
        edge_lengths_flattened[ancestry[i, 1]] = edge_lengths[i, 1]

    return v, edge_lengths_flattened


def beagle_jc69_loss(m, data, model):
    v, edgeLengths = parse_matrix(m)

    nPatterns = len(data[0])
    nEdges = 2 * len(m)

    beagle_instance = Beagle4State(
        n_tips=len(m) + 1,
        n_internal=len(m),
        n_buffers=len(m) + 1,
        n_states=4,
        n_patterns=nPatterns,
        n_eigendecomposition_buffers=1,
        n_transition_buffers=nEdges,
        n_rate_cats=1,
        n_scaling_buffers=0,
    )

    states = [createStates(x) for x in data]

    for i, s in enumerate(states):
        beagle_instance.setTipStates(i, s)

    patternWeights = np.ones((len(states[0]),))
    beagle_instance.setPatternWeights(patternWeights)

    # State background frequencies
    freqs = np.full((4,), 0.25)
    beagle_instance.setStateFrequencies(0, freqs)

    # Category weights and rates
    weights = np.array([1.0])
    rates = np.array([1.0])
    beagle_instance.setCategoryWeights(0, weights)
    beagle_instance.setCategoryRates(rates)

    # Eigendecomposition for JC69
    beagle_instance.setEigenDecomposition(0, model)

    # Node indices
    nodeIndices = np.arange(0, nEdges, dtype=np.int64)
    beagle_instance.updateTransitionMatrices(0, nodeIndices, edgeLengths, nEdges)

    operations = vToOperations(v)
    beagle_instance.updatePartials(operations, operations.shape[0], BEAGLE_OP_NONE)

    rootIndex = [2 * len(v)]
    categoryWeightIndex = np.array([0])
    stateFrequencyIndex = np.array([0])
    cumulativeScaleIndex = [BEAGLE_OP_NONE]

    return beagle_instance.calcRootLogLikelihoods(
        rootIndex[0],
        categoryWeightIndex[0],
        stateFrequencyIndex[0],
        cumulativeScaleIndex[0],
    )
