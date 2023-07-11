/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_PODINPUTOUTPUTTYPES_H
#define TUDAT_PODINPUTOUTPUTTYPES_H

#include <map>
#include <vector>
#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <Eigen/LU>

#include "tudat/basics/timeType.h"
#include "tudat/astro/observation_models/linkTypeDefs.h"
#include "tudat/astro/observation_models/observableTypes.h"
#include "tudat/simulation/estimation_setup/observations.h"
#include "tudat/simulation/propagation_setup/dynamicsSimulator.h"

namespace tudat
{

namespace simulation_setup
{

template< typename ObservationScalarType = double, typename TimeType = double >
class CovarianceAnalysisInput
{
public:
    CovarianceAnalysisInput(
              const std::shared_ptr< observation_models::ObservationCollection< ObservationScalarType, TimeType > >& observationCollection,
              const Eigen::MatrixXd inverseOfAprioriCovariance = Eigen::MatrixXd::Zero( 0, 0 ) ):
        observationCollection_( observationCollection ),
        inverseOfAprioriCovariance_( inverseOfAprioriCovariance ),
        reintegrateEquationsOnFirstIteration_( true ),
        reintegrateVariationalEquations_( true ),
        saveDesignMatrix_( true ),
        printOutput_( true )
    {
        weightsMatrixDiagonals_ = Eigen::VectorXd::Zero( observationCollection->getTotalObservableSize( ) );
        setConstantWeightsMatrix( 1.0 );
    }

    virtual ~CovarianceAnalysisInput( ){ }

    //! Function to set a constant values for all observation weights
    /*!
     * Function to set a constant values for all observation weights
     * \param constantWeight Constant weight that is to be set for all observations
     */
    void setConstantWeightsMatrix( const double constantWeight = 1.0 )
    {
        weightsMatrixDiagonals_.setConstant(
                    observationCollection_->getTotalObservableSize( ), constantWeight );
    }

    //! Function to set a values for observation weights, constant per observable type
    /*!
     * Function to set a values for observation weights, constant per observable type
     * \param weightPerObservable Values for observation weights, constant per observable type
     */
    void setConstantPerObservableWeightsMatrix(
            const std::map< observation_models::ObservableType, double > weightPerObservable )
    {
        std::map< observation_models::ObservableType, std::pair< int, int > > observationTypeStartAndSize =
                observationCollection_->getObservationTypeStartAndSize( );

        for( auto observableIterator : weightPerObservable )
        {
            observation_models::ObservableType currentObservable = observableIterator.first;
            if( observationTypeStartAndSize.count( observableIterator.first ) == 0 )
            {
                std::cerr<<"Warning when setting weights for data type " <<std::to_string( observableIterator.first )<< ". "<<
                           " No data of given type found."<<std::endl;
            }
            else
            {
                weightsMatrixDiagonals_.segment( observationTypeStartAndSize.at( currentObservable ).first,
                                                 observationTypeStartAndSize.at( currentObservable ).second ) =
                        Eigen::VectorXd::Constant( observationTypeStartAndSize.at( currentObservable ).second, observableIterator.second );
            }

        }
    }

    //! Function to set a values for observation weights, constant per observable type and link ends type
    /*!
     * Function to set a values for observation weights, constant per observable type and link ends type
     * \param weightPerObservableAndLinkEnds Values for observation weights, constant per observable type and link ends type
     */
    void setConstantPerObservableAndLinkEndsWeights(
            const std::map< observation_models::ObservableType,
            std::map< observation_models::LinkEnds, double > > weightPerObservableAndLinkEnds )
    {
        std::map< observation_models::ObservableType,
                std::map< observation_models::LinkEnds, std::vector< std::pair< int, int > > > >  observationSetStartAndSize =
                observationCollection_->getObservationSetStartAndSize( );

        for( auto observableIterator : weightPerObservableAndLinkEnds )
        {
            observation_models::ObservableType currentObservable = observableIterator.first;
            if( observationSetStartAndSize.count( currentObservable) == 0 )
            {
                std::cerr<< "Warning when setting weights for data type "<< std::to_string( currentObservable) <<  ". " <<
                            " No data of given type found." <<std::endl;
            }
            else
            {
                for( auto linkEndIterator : observableIterator.second )
                {
                    observation_models::LinkEnds currentLinkEnds = linkEndIterator.first;
                    if( observationSetStartAndSize.at( currentObservable ).count( currentLinkEnds ) == 0 )
                    {

                        std::cerr<< "Warning when setting weights for data type " << std::to_string( currentObservable)<< " and link ends " <<
                                    //static_cast< std::string >( currentLinkEnds ) +
                                    ". No data of given type and link ends found." <<std::endl;
                    }
                    else
                    {
                        std::vector< std::pair< int, int > > indicesToUse =
                                observationSetStartAndSize.at( currentObservable ).at( currentLinkEnds );
                        for( unsigned int i = 0; i < indicesToUse.size( ); i++ )
                        {
                            weightsMatrixDiagonals_.segment( indicesToUse.at( i ).first,
                                                             indicesToUse.at( i ).second ) =
                                    Eigen::VectorXd::Constant( indicesToUse.at( i ).second, linkEndIterator.second );
                        }
                    }
                }
            }

        }
    }
    void setConstantPerObservableAndLinkEndsWeights(
            const observation_models::ObservableType observableType,
            const std::vector< observation_models::LinkEnds >& linkEnds,
            const double weight )
    {
        std::map< observation_models::ObservableType, std::map< observation_models::LinkEnds, double > > weightPerObservableAndLinkEnds;
        for( unsigned int i = 0; i < linkEnds.size( ); i++ )
        {
            weightPerObservableAndLinkEnds[ observableType ][ linkEnds.at( i ) ] =  weight;
        }
        setConstantPerObservableAndLinkEndsWeights( weightPerObservableAndLinkEnds );
    }


    void setConstantPerObservableAndLinkEndsWeights(
            const observation_models::ObservableType observableType,
            const observation_models::LinkEnds& linkEnds,
            const double weight )
    {
        std::map< observation_models::ObservableType, std::map< observation_models::LinkEnds, double > > weightPerObservableAndLinkEnds;
        weightPerObservableAndLinkEnds[ observableType ][ linkEnds ] =  weight;
        setConstantPerObservableAndLinkEndsWeights( weightPerObservableAndLinkEnds );
    }

    //! Function to return the total data structure of observations and associated times/link ends/type (by reference)
    /*!
     * Function to return the total data structure of observations and associated times/link ends/type (by reference)
     * \return Total data structure of observations and associated times/link ends/type (by reference)
     */
    std::shared_ptr< observation_models::ObservationCollection< ObservationScalarType, TimeType > > getObservationCollection( )
    {
        return observationCollection_;
    }

    //! A priori covariance matrix (unnormalized) of estimated parameters
    //! Function to return the a priori covariance matrix (unnormalized) of estimated parameters
    /*!
     * Function to return the a priori covariance matrix (unnormalized) of estimated parameters
     * \return A priori covariance matrix (unnormalized) of estimated parameters
     */
    Eigen::MatrixXd getInverseOfAprioriCovariance( )
    {
        return inverseOfAprioriCovariance_;
    }

    Eigen::MatrixXd getInverseOfAprioriCovariance( const int numberOfParameters )
    {
        if( inverseOfAprioriCovariance_.rows( ) == 0 )
        {
            return Eigen::MatrixXd::Zero( numberOfParameters, numberOfParameters );
        }
        else if( inverseOfAprioriCovariance_.rows( ) != numberOfParameters ||
                 inverseOfAprioriCovariance_.cols( ) != numberOfParameters )
        {
            throw std::runtime_error( "Error whenr retrieving invers a priori covariance; size is incompatible" );
        }
        else
        {
            return inverseOfAprioriCovariance_;
        }
    }


    //! Function to return the weight matrix diagonals, sorted by link ends and observable type (by reference)
    /*!
     * Function to return the weight matrix diagonals, sorted by link ends and observable type (by reference)
     * \return Weight matrix diagonals, sorted by link ends and observable type (by reference)
     */
    Eigen::VectorXd getWeightsMatrixDiagonals( )
    {
        return weightsMatrixDiagonals_;
    }

    //! Function to return the boolean denoting whether the dynamics and variational equations are reintegrated on first iteration
    /*!
     * Function to return the boolean denoting whether the dynamics and variational equations are to be reintegrated on first
     * iteration
     * \return Boolean denoting whether the dynamics and variational equations are to be reintegrated on first iteration
     */
    bool getReintegrateEquationsOnFirstIteration( )
    {
        return reintegrateEquationsOnFirstIteration_;
    }

    //! Function to return the boolean denoting whether the variational equations are to be reintegrated during estimation
    /*!
     * Function to return the boolean denoting whether the variational equations are to be reintegrated during estimation
     * \return Boolean denoting whether the variational equations are to be reintegrated during estimation
     */
    bool getReintegrateVariationalEquations( )
    {
        return reintegrateVariationalEquations_;
    }

    //! Function to return the boolean denoting whether to print output to th terminal when running the estimation.
    /*!
     * Function to return the boolean denoting whether to print output to th terminal when running the estimation.
     * \return Boolean denoting whether to print output to th terminal when running the estimation.
     */
    bool getSaveDesignMatrix( )
    {
        return saveDesignMatrix_;
    }

    //! Function to return the boolean denoting whether to print output to th terminal when running the estimation.
    /*!
     * Function to return the boolean denoting whether to print output to th terminal when running the estimation.
     * \return Boolean denoting whether to print output to th terminal when running the estimation.
     */
    bool getPrintOutput( )
    {
        return printOutput_;
    }

    void defineCovarianceSettings( const bool reintegrateEquationsOnFirstIteration = 1,
                                   const bool reintegrateVariationalEquations = 1,
                                   const bool saveDesignMatrix = 1,
                                   const bool printOutput = 1 )
    {
        this->reintegrateEquationsOnFirstIteration_ = reintegrateEquationsOnFirstIteration;
        this->reintegrateVariationalEquations_ = reintegrateVariationalEquations;
        this->saveDesignMatrix_ = saveDesignMatrix;
        this->printOutput_ = printOutput;
    }



protected:
    //! Total data structure of observations and associated times/link ends/type
    std::shared_ptr< observation_models::ObservationCollection< ObservationScalarType, TimeType > > observationCollection_;

    //! A priori covariance matrix (unnormalized) of estimated parameters
    Eigen::MatrixXd inverseOfAprioriCovariance_;

    //! Weight matrix diagonals, sorted by link ends and observable type
    Eigen::VectorXd weightsMatrixDiagonals_;

    //!  Boolean denoting whether the dynamics and variational equations are to be reintegrated on first iteration
    bool reintegrateEquationsOnFirstIteration_;

    //! Boolean denoting whether the variational equations are to be reintegrated during estimation
    bool reintegrateVariationalEquations_;

    //! Boolean denoting whether to print output to th terminal when running the estimation.
    bool saveDesignMatrix_;

    //! Boolean denoting whether to print output to th terminal when running the estimation.
    bool printOutput_;
};


//! Class that is used during the orbit determination/parameter estimation to determine whether the estimation is converged.
class EstimationConvergenceChecker
{
public:

    //! Constructor
    /*!
     * Constructor, sets a number of values for stopping conditions. The estimation stops if one of these is met.
     * \param maximumNumberOfIterations Maximum number of allowed iterations for estimation
     * \param minimumResidualChange Minimum required change in residual between two iterations
     * \param minimumResidual Minimum value of observation residual below which estimation is converged
     * \param numberOfIterationsWithoutImprovement Number of iterations without reduction of residual
     */
    EstimationConvergenceChecker(
            const unsigned int maximumNumberOfIterations = 5,
            const double minimumResidualChange = 0.0,
            const double minimumResidual = 1.0E-20,
            const int numberOfIterationsWithoutImprovement = 2 ):
        maximumNumberOfIterations_( maximumNumberOfIterations ), minimumResidualChange_( minimumResidualChange ),
        minimumResidual_( minimumResidual ),
        numberOfIterationsWithoutImprovement_( numberOfIterationsWithoutImprovement )
    { }

    //! Function to determine whether the estimation is deemed to be converged
    /*!
     * Function to determine whether the estimation is deemed to be converged (i.e. if it should terminate)
     * \param numberOfIterations Number of iterations of estimation procedure that have been completed
     * \param rmsResidualHistory Rms residuals at current and all previous iterations
     * \return True if estimation is to be terminated
     */
    bool isEstimationConverged( const int numberOfIterations, const std::vector< double > rmsResidualHistory )
    {
        bool isConverged = 0;
        if( numberOfIterations >= maximumNumberOfIterations_ )
        {
            std::cout << "Maximum number of iterations reached" << std::endl;
            isConverged = 1;
        }
        if( rmsResidualHistory[ rmsResidualHistory.size( ) - 1 ] < minimumResidual_ )
        {
            std::cout << "Required residual level achieved" << std::endl;
            isConverged = 1;
        }
        if( ( std::distance( rmsResidualHistory.begin( ), std::max_element(
                                 rmsResidualHistory.begin( ), rmsResidualHistory.end( ) ) ) - rmsResidualHistory.size( ) ) <
                numberOfIterationsWithoutImprovement_ )
        {
            std::cout << "Too many iterations without parameter improvement" << std::endl;
            isConverged = 1;
        }
        if( rmsResidualHistory.size( ) > 1 )
        {
            if( std::fabs( rmsResidualHistory.at( rmsResidualHistory.size( )  - 1 ) -
                           rmsResidualHistory.at( rmsResidualHistory.size( )  - 2 ) ) < minimumResidualChange_ )
            {
                isConverged = 1;
            }
        }
        return isConverged;
    }


protected:

    //! Maximum number of allowed iterations for estimation
    int maximumNumberOfIterations_;

    //! Minimum required change in residual between two iterations
    double minimumResidualChange_;

    //! Minimum value of observation residual below which estimation is converged
    double minimumResidual_;

    //!  Number of iterations without reduction of residual
    unsigned int numberOfIterationsWithoutImprovement_;
};



//! Data structure used to provide input to orbit determination procedure
template< typename ObservationScalarType = double, typename TimeType = double >
class EstimationInput: public CovarianceAnalysisInput< ObservationScalarType, TimeType >
{
public:

    //! Constructor
    /*!
     * Constructor
     * \param observationCollection Total data structure of observations and associated times/link ends/type
     * \param inverseOfAprioriCovariance A priori covariance matrix (unnormalized) of estimated parameters. None (matrix of
     * size 0) by default
     */
    EstimationInput(
            const std::shared_ptr< observation_models::ObservationCollection< ObservationScalarType, TimeType > >& observationCollection,
            const Eigen::MatrixXd inverseOfAprioriCovariance = Eigen::MatrixXd::Zero( 0, 0 ),
            const std::shared_ptr< EstimationConvergenceChecker > convergenceChecker =
            std::make_shared< EstimationConvergenceChecker >( ) ):
        CovarianceAnalysisInput< ObservationScalarType, TimeType >( observationCollection, inverseOfAprioriCovariance ),
        saveResidualsAndParametersFromEachIteration_( true ),
        saveStateHistoryForEachIteration_( false ),
        convergenceChecker_( convergenceChecker )
    { }

    //! Destructor
    virtual ~EstimationInput( ){ }


    //! Function to define specific settings for estimation process
    /*!
     *  Function to define specific settings for estimation process
     *  \param reintegrateEquationsOnFirstIteration Boolean denoting whether the dynamics and variational equations are to
     *  be reintegrated on first iteration, or if existing values are to be used to perform first iteration.
     *  \param reintegrateVariationalEquations Boolean denoting whether the variational equations are to be reintegrated during
     *  estimation
     *  \param saveDesignMatrix Boolean denoting whether to save the partials matrix in the output
     *  \param printOutput Boolean denoting whether to print output to th terminal when running the estimation.
     *  \param saveResidualsAndParametersFromEachIteration Boolean denoting whether the residuals and parameters from the each
     *  iteration are to be saved
     *  \param saveStateHistoryForEachIteration Boolean denoting whether the state history is to be saved on each iteration
     */
    void defineEstimationSettings( const bool reintegrateEquationsOnFirstIteration = 1,
                                   const bool reintegrateVariationalEquations = 1,
                                   const bool saveDesignMatrix = 1,
                                   const bool printOutput = 1,
                                   const bool saveResidualsAndParametersFromEachIteration = 1,
                                   const bool saveStateHistoryForEachIteration = 0 )
    {
        this->reintegrateEquationsOnFirstIteration_ = reintegrateEquationsOnFirstIteration;
        this->reintegrateVariationalEquations_ = reintegrateVariationalEquations;
        this->saveDesignMatrix_ = saveDesignMatrix;
        this->printOutput_ = printOutput;
        this->saveResidualsAndParametersFromEachIteration_ = saveResidualsAndParametersFromEachIteration;
        this->saveStateHistoryForEachIteration_ = saveStateHistoryForEachIteration;
    }


    //! Function to return the boolean denoting whether the residuals and parameters from the each iteration are to be saved
    /*!
     * Function to return the boolean denoting whether the residuals and parameters from the each iteration are to be saved
     * \return Boolean denoting whether the residuals and parameters from the each iteration are to be saved
     */
    bool getSaveResidualsAndParametersFromEachIteration( )
    {
        return saveResidualsAndParametersFromEachIteration_;
    }


    std::shared_ptr< EstimationConvergenceChecker > getConvergenceChecker( )
    {
        return convergenceChecker_;
    }


    void setConvergenceChecker( const std::shared_ptr< EstimationConvergenceChecker > convergenceChecker )
    {
        convergenceChecker_ = convergenceChecker;
    }


    //! Function to return the boolean denoting whether the state history is to be saved on each iteration.
    /*!
     * Function to return the boolean denoting whether the state history is to be saved on each iteration.
     * \return Boolean denoting whether the state history is to be saved on each iteration.
     */
    bool getSaveStateHistoryForEachIteration( )
    {
        return saveStateHistoryForEachIteration_;
    }

    //! Boolean denoting whether the residuals and parameters from the each iteration are to be saved
    bool saveResidualsAndParametersFromEachIteration_;

    //! Boolean denoting whether the state history is to be saved on each iteration.
    bool saveStateHistoryForEachIteration_;

    std::shared_ptr< EstimationConvergenceChecker > convergenceChecker_;    


};

inline std::shared_ptr< EstimationConvergenceChecker > estimationConvergenceChecker(
        const unsigned int maximumNumberOfIterations = 5,
        const double minimumResidualChange = 0.0,
        const double minimumResidual = 1.0E-20,
        const int numberOfIterationsWithoutImprovement = 2 )
{
    return std::make_shared< EstimationConvergenceChecker >(
                maximumNumberOfIterations, minimumResidualChange, minimumResidual, numberOfIterationsWithoutImprovement );
}


void scaleDesignMatrixWithWeights(
        Eigen::MatrixXd& designMatrix,
        const Eigen::VectorXd& weightsDiagonal );


template< typename ObservationScalarType = double, typename TimeType = double  >
struct CovarianceAnalysisOutput
{

    CovarianceAnalysisOutput( const Eigen::MatrixXd& normalizedDesignMatrix,
                             const Eigen::VectorXd& weightsMatrixDiagonal,
                             const Eigen::VectorXd& designMatrixTransformationDiagonal,
                             const Eigen::MatrixXd& inverseNormalizedCovarianceMatrix,
                             const bool exceptionDuringPropagation = false ):
        normalizedDesignMatrix_( normalizedDesignMatrix ),
        weightsMatrixDiagonal_( weightsMatrixDiagonal ),
        designMatrixTransformationDiagonal_( designMatrixTransformationDiagonal ),
        inverseNormalizedCovarianceMatrix_( inverseNormalizedCovarianceMatrix ),
        exceptionDuringPropagation_( exceptionDuringPropagation )
    { }


    Eigen::VectorXd getNormalizationTerms( )
    {
        return designMatrixTransformationDiagonal_;
    }

    Eigen::MatrixXd getNormalizedInverseCovarianceMatrix( )
    {
        return inverseNormalizedCovarianceMatrix_;
    }

    //! Function to retrieve the unnormalized inverse estimation covariance matrix
    /*!
     * Function to retrieve the unnormalized inverse estimation covariance matrix
     * \return Isnverse estimation covariance matrix
     */
    Eigen::MatrixXd getUnnormalizedInverseCovarianceMatrix( )
    {

        Eigen::MatrixXd inverseUnnormalizedCovarianceMatrix = getNormalizedInverseCovarianceMatrix( );

        for( int i = 0; i < designMatrixTransformationDiagonal_.rows( ); i++ )
        {
            for( int j = 0; j < designMatrixTransformationDiagonal_.rows( ); j++ )
            {
                inverseUnnormalizedCovarianceMatrix( i, j ) *=
                        designMatrixTransformationDiagonal_( i ) * designMatrixTransformationDiagonal_( j );
            }
        }

        return inverseUnnormalizedCovarianceMatrix;
    }

    Eigen::MatrixXd getNormalizedCovarianceMatrix( )
    {
        return inverseNormalizedCovarianceMatrix_.inverse( );
    }

    //! Function to retrieve the unnormalized estimation covariance matrix
    /*!
     * Function to retrieve the unnormalized estimation covariance matrix
     * \return estimation covariance matrix
     */
    Eigen::MatrixXd getUnnormalizedCovarianceMatrix( )
    {
        Eigen::MatrixXd unnormalizedCovarianceMatrix = getNormalizedCovarianceMatrix( );

        for( int i = 0; i < designMatrixTransformationDiagonal_.rows( ); i++ )
        {
            for( int j = 0; j < designMatrixTransformationDiagonal_.rows( ); j++ )
            {
                unnormalizedCovarianceMatrix( i, j ) /=
                        designMatrixTransformationDiagonal_( i ) * designMatrixTransformationDiagonal_( j );
            }
        }

        return unnormalizedCovarianceMatrix;
    }

    //! Function to retrieve the matrix of unnormalized partial derivatives
    /*!
     * Function to retrieve the matrix of unnormalized partial derivatives (typically detnoed as H)
     * \return Matrix of unnormalized partial derivatives
     */
    Eigen::MatrixXd getUnnormalizedDesignMatrix( )
    {
        Eigen::MatrixXd unnormalizedPartialDerivatives = Eigen::MatrixXd::Zero(
                    normalizedDesignMatrix_.rows( ), normalizedDesignMatrix_.cols( ) );

        for( int i = 0; i < designMatrixTransformationDiagonal_.rows( ); i++ )
        {
            unnormalizedPartialDerivatives.block( 0, i, normalizedDesignMatrix_.rows( ), 1 ) =
                    normalizedDesignMatrix_.block( 0, i, normalizedDesignMatrix_.rows( ), 1 ) *
                    designMatrixTransformationDiagonal_( i );
        }
        return unnormalizedPartialDerivatives;
    }

    Eigen::MatrixXd getNormalizedDesignMatrix( )
    {
        return normalizedDesignMatrix_;
    }


    Eigen::MatrixXd getNormalizedWeightedDesignMatrix( )
    {
        Eigen::MatrixXd weightedNormalizedDesignMatrix = normalizedDesignMatrix_;
        scaleDesignMatrixWithWeights(
                    weightedNormalizedDesignMatrix,
                    weightsMatrixDiagonal_ );
        return weightedNormalizedDesignMatrix;
    }

    Eigen::MatrixXd getUnnormalizedWeightedDesignMatrix( )
    {
        Eigen::MatrixXd weightedUnnormalizedDesignMatrix = getUnnormalizedDesignMatrix( );
        scaleDesignMatrixWithWeights(
                    weightedUnnormalizedDesignMatrix,
                    weightsMatrixDiagonal_ );
        return weightedUnnormalizedDesignMatrix;
    }


    //! Function to retrieve the unnormalized formal error vector of the estimation result.
    /*!
     * Function to retrieve the unnormalized formal error vector of the estimation result.
     * \return Formal error vector of the estimation result.
     */
    Eigen::VectorXd getFormalErrorVector( )
    {
        return ( getUnnormalizedCovarianceMatrix( ).diagonal( ) ).cwiseSqrt( );
    }

    //! Function to retrieve the correlation matrix of the estimation result.
    /*!
     * Function to retrieve the correlation matrix of the estimation result.
     * \return Correlation matrix of the estimation result.
     */
    Eigen::MatrixXd getCorrelationMatrix( )
    {
        return getUnnormalizedCovarianceMatrix( ).cwiseQuotient( getFormalErrorVector( ) * getFormalErrorVector( ).transpose( ) );
    }

    //! Matrix of observation partials (normalixed) used in estimation (may be empty if so requested)
    Eigen::MatrixXd normalizedDesignMatrix_;

    //! Diagonal of weights matrix used in the estimation
    Eigen::VectorXd weightsMatrixDiagonal_;

    //! Vector of values by which the columns of the unnormalized information matrix were divided to normalize its entries.
    Eigen::VectorXd designMatrixTransformationDiagonal_;

    //! Inverse of postfit normalized covariance matrix
    Eigen::MatrixXd inverseNormalizedCovarianceMatrix_;

    //! Boolean denoting whether an exception was caught during (re)propagation of equations of motion (and variational equations)
    bool exceptionDuringPropagation_;
};

//! Data structure through which the output of the orbit determination is communicated
template< typename ObservationScalarType = double, typename TimeType = double  >
struct EstimationOutput: public CovarianceAnalysisOutput< ObservationScalarType, TimeType >
{

    //! Constructor
    /*!
     * Constructor
     * \param parameterEstimate Vector of estimated parameter values.
     * \param residuals Vector of postfit observation residuals
     * \param normalizedDesignMatrix Matrix of observation partials (normalixed) used in estimation
     * (may be empty if so requested)
     * \param weightsMatrixDiagonal Diagonal of weights matrix used in the estimation
     * \param designMatrixTransformationDiagonal Vector of values by which the columns of the unnormalized information
     * matrix were divided to normalize its entries.
     * \param inverseNormalizedCovarianceMatrix Inverse of postfit normalized covariance matrix
     * \param residualStandardDeviation Standard deviation of postfit residuals vector
     * \param residualHistory Vector of residuals per iteration
     * \param parameterHistory Vector of parameter vectors per iteration (entry 1 is pre-estimation values)
     * \param exceptionDuringInversion Boolean denoting whether an exception was caught during inversion of normal equations
     * \param exceptionDuringPropagation Boolean denoting whether an exception was caught during (re)propagation of equations of
     * motion (and variational equations).
     */
    EstimationOutput( const Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, 1 >& parameterEstimate,
               const Eigen::VectorXd& residuals,
               const Eigen::MatrixXd& normalizedDesignMatrix,
               const Eigen::VectorXd& weightsMatrixDiagonal,
               const Eigen::VectorXd& designMatrixTransformationDiagonal,
               const Eigen::MatrixXd& inverseNormalizedCovarianceMatrix,
               const double residualStandardDeviation,
               const int bestIteration,
               const std::vector< Eigen::VectorXd >& residualHistory = std::vector< Eigen::VectorXd >( ),
               const std::vector< Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, 1 > >& parameterHistory = std::vector< Eigen::VectorXd >( ),
               const bool exceptionDuringInversion = false,
               const bool exceptionDuringPropagation = false ):
        CovarianceAnalysisOutput< ObservationScalarType, TimeType >( normalizedDesignMatrix, weightsMatrixDiagonal,
                                 designMatrixTransformationDiagonal, inverseNormalizedCovarianceMatrix,
                                 exceptionDuringPropagation ),
        parameterEstimate_( parameterEstimate ),
        residuals_( residuals ),
        bestIteration_( bestIteration ),
        residualStandardDeviation_( residualStandardDeviation ),
        residualHistory_( residualHistory ),
        parameterHistory_( parameterHistory ),
        exceptionDuringInversion_( exceptionDuringInversion ),
        numberOfParameters_( normalizedDesignMatrix.cols( ) )
    { }


    //! Function to get residual vectors per iteration concatenated into a matrix
    /*!
     * Function to get residual vectors per iteration concatenated into a matrix (one column per iteration).
     * \return Residual vectors per iteration concatenated into a matrix
     */
    Eigen::MatrixXd getResidualHistoryMatrix( )
    {
        if( residualHistory_.size( ) > 0 )
        {
            Eigen::MatrixXd residualHistoryMatrix = Eigen::MatrixXd( residualHistory_.at( 0 ).rows( ), residualHistory_.size( ) );
            for( unsigned int i = 0; i < residualHistory_.size( ); i++ )
            {
                residualHistoryMatrix.block( 0, i, residualHistory_.at( 0 ).rows( ), 1 ) = residualHistory_.at( i );
            }
            return residualHistoryMatrix;
        }
        else
        {
            std::cerr << "Warning, requesting residual history, but history not saved." << std::endl;
            return Eigen::MatrixXd::Zero( 0, 0 );
        }
    }

    //! Function to get parameter vectors per iteration concatenated into a matrix
    /*!
     * Function to get parameter vectors per iteration concatenated into a matrix (one column per iteration). Column 0 contains
     * pre-estimation values
     * \return Parameter vectors per iteration concatenated into a matrix
     */
    Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, Eigen::Dynamic > getParameterHistoryMatrix( )
    {
        if( parameterHistory_.size( ) > 0 )
        {
            Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, Eigen::Dynamic > parameterHistoryMatrix =
                    Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, Eigen::Dynamic >(
                        parameterHistory_.at( 0 ).rows( ), parameterHistory_.size( ) );
            for( unsigned int i = 0; i < parameterHistory_.size( ); i++ )
            {
                parameterHistoryMatrix.block( 0, i, parameterHistory_.at( 0 ).rows( ), 1 ) = parameterHistory_.at( i );
            }
            return parameterHistoryMatrix;
        }
        else
        {
            std::cerr << "Warning, requesting parameter history, but history not saved." << std::endl;
            return Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, Eigen::Dynamic >::Zero( 0, 0 );
        }
    }


//    std::vector< std::vector< std::map< TimeType, Eigen::VectorXd > > > getDependentVariableHistory( )
//    {
//        if( dependentVariableHistoryPerIteration_.size( ) == 0 )
//        {
//            throw std::runtime_error( "Error when retrieving dependent variable histories from estimation, no dependent variable histories set." );
//        }
//        return dependentVariableHistoryPerIteration_;
//    }

//    std::vector< std::vector< std::map< TimeType, Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, 1 > > > > getDynamicsHistory( )
//    {
//        if( dynamicsHistoryPerIteration_.size( ) == 0 )
//        {
//            throw std::runtime_error( "Error when retrieving dynamics histories from estimation, no dynamics results set." );
//        }
//        return dynamicsHistoryPerIteration_;
//    }

//    std::vector< std::map< TimeType, Eigen::VectorXd > > getFinalDependentVariableHistory( )
//    {
//        if( dependentVariableHistoryPerIteration_.size( ) == 0 )
//        {
//            throw std::runtime_error( "Error when retrieving final dependent variable histories from estimation, no dependent variable histories set." );
//        }
//        return dependentVariableHistoryPerIteration_.at( dependentVariableHistoryPerIteration_.size( ) - 1 );
//    }

//    std::vector< std::map< TimeType, Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, 1 > > > getFinalDynamicsHistory( )
//    {
//        if( dynamicsHistoryPerIteration_.size( ) == 0 )
//        {
//            throw std::runtime_error( "Error when retrieving final dynamics histories from estimation, no dynamics results set." );
//        }
//        return dynamicsHistoryPerIteration_.at( dynamicsHistoryPerIteration_.size( ) - 1 );
//    }

    void setSimulationResults(
            std::vector< std::shared_ptr< propagators::SimulationResults< ObservationScalarType, TimeType > > > simulationResultsPerIteration )
    {
        simulationResultsPerIteration_ = simulationResultsPerIteration;
    }

    std::vector< std::shared_ptr< propagators::SimulationResults< ObservationScalarType, TimeType > > >  getSimulationResults( )
    {
        return simulationResultsPerIteration_;
    }

    std::shared_ptr< propagators::SimulationResults< ObservationScalarType, TimeType > > getBestIterationSimulationResults( )
    {
        return simulationResultsPerIteration_.at( bestIteration_ );
    }


    //! Vector of estimated parameter values.
    Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, 1 > parameterEstimate_;

    //! Vector of postfit observation residuals
    Eigen::VectorXd residuals_;

    int bestIteration_;

    //! Standard deviation of postfit residuals vector
    double residualStandardDeviation_;

    //! Vector of residuals per iteration
    std::vector< Eigen::VectorXd > residualHistory_;

    //! Vector of parameter vectors per iteration (entry 0 is pre-estimation values)
    std::vector< Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, 1 > > parameterHistory_;

    //! Boolean denoting whether an exception was caught during inversion of normal equations
    bool exceptionDuringInversion_;

    int numberOfParameters_;


    std::vector< std::shared_ptr< propagators::SimulationResults< ObservationScalarType, TimeType > > > simulationResultsPerIteration_;

//    //! List of numerical solutions of dynamics (per iteration, per arc)
//    std::vector< std::vector< std::map< TimeType, Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, 1 > > > > dynamicsHistoryPerIteration_;

//    //! List of numerical solutions of dependent variables (per iteration, per arc)
//    std::vector< std::vector< std::map< TimeType, Eigen::VectorXd > > > dependentVariableHistoryPerIteration_;





};


extern template class EstimationInput< double, double >;
extern template struct EstimationOutput< double >;

}

}

#endif // TUDAT_PODINPUTOUTPUTTYPES_H
