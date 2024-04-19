/*    Copyright (c) 2010-2022, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include "tudat/astro/electromagnetism/reflectionLaw.h"

#include <Eigen/Core>

#include "tudat/math/basic/linearAlgebra.h"
#include "tudat/math/basic/mathematicalConstants.h"


namespace tudat
{
namespace electromagnetism
{

double SpecularDiffuseMixReflectionLaw::evaluateReflectedFraction(const Eigen::Vector3d& surfaceNormal,
                                                                  const Eigen::Vector3d& incomingDirection,
                                                                  const Eigen::Vector3d& observerDirection) const
{
    // Check if any reflected radiation would reach observer
    const double cosBetweenNormalAndIncoming = surfaceNormal.dot(-incomingDirection);
    const double cosBetweenNormalAndObserver = surfaceNormal.dot(observerDirection);
    if (cosBetweenNormalAndIncoming <= 0 || cosBetweenNormalAndObserver <= 0)
    {
        // Radiation is incident on backside, or observer is on backside
        return 0;
    }

    // Wetterer (2014) Eq. 4
    const auto diffuseReflectance = diffuseReflectivity_ / mathematical_constants::PI;

    double instantaneousReradiationReflectance = 0;
    if (withInstantaneousReradiation_)
    {
        instantaneousReradiationReflectance = absorptivity_ / mathematical_constants::PI;
    }

    double specularReflectance = 0;
    if (specularReflectivity_ > 0) {
        const auto mirrorOfIncomingDirection = computeMirrorlikeReflection(incomingDirection, surfaceNormal);
        if (observerDirection.isApprox(mirrorOfIncomingDirection))
        {
            // Observer only receives specular reflection if it is in mirrored path of incident radiation
            // Wetterer (2014) Eq. 4
            specularReflectance = specularReflectivity_ / cosBetweenNormalAndIncoming;
        }
    }
    
    return diffuseReflectance + specularReflectance + instantaneousReradiationReflectance;
}

Eigen::Vector3d SpecularDiffuseMixReflectionLaw::evaluateReactionVector(const Eigen::Vector3d& surfaceNormal,
                                                                        const Eigen::Vector3d& incomingDirection) const
{
    const double cosBetweenNormalAndIncoming = surfaceNormal.dot(-incomingDirection);
    if (cosBetweenNormalAndIncoming <= 0)
    {
        // Radiation is incident on backside of surface
        return Eigen::Vector3d::Zero();
    }

    // Montenbruck (2014) Eq. 5
    // Use auto here to force single lazy evaluation upon return
    auto reactionFromIncidence = (absorptivity_ + diffuseReflectivity_) * incomingDirection;
    auto reactionFromReflection =
            -(2. / 3 * diffuseReflectivity_ + 2 * specularReflectivity_ * cosBetweenNormalAndIncoming) * surfaceNormal;

    Eigen::Vector3d reactionFromInstantaneousReradiation;
    if (withInstantaneousReradiation_)
    {
        // Montenbruck (2014) Eq. 6
        // Instantaneous Lambertian reradiation behaves like diffuse Lambertian reflection
        reactionFromInstantaneousReradiation = -(2. / 3 * absorptivity_) * surfaceNormal;
    }
    else
    {
        reactionFromInstantaneousReradiation = Eigen::Vector3d::Zero();
    }

    return reactionFromIncidence + reactionFromReflection + reactionFromInstantaneousReradiation;
}

Eigen::Matrix3d SpecularDiffuseMixReflectionLaw::evaluateReactionVectorDerivativeWrtTargetPosition(
    const Eigen::Vector3d& surfaceNormal,
    const Eigen::Vector3d& incomingDirection,
    const double cosineOfAngleBetweenVectors,
    const Eigen::Vector3d& currentReactionVector,
    const Eigen::Matrix3d& sourceUnitVectorPartial,
    const Eigen::Matrix< double, 1, 3 >& cosineAnglePartial )
{
    return currentReactionVector * cosineAnglePartial / cosineOfAngleBetweenVectors -
           cosineOfAngleBetweenVectors * ( (absorptivity_ + diffuseReflectivity_) * sourceUnitVectorPartial +
                                           2.0 * specularReflectivity_ * surfaceNormal * cosineAnglePartial );
}

// Solar Sail reflection law
double SolarSailOpticalReflectionLaw::evaluateReflectedFraction(const Eigen::Vector3d& surfaceNormal,
                                                                  const Eigen::Vector3d& incomingDirection,
                                                                  const Eigen::Vector3d& observerDirection) const
{
    throw std::runtime_error(
            "Error. This functionality is not available for the solar sail optical model.");
}

Eigen::Vector3d SolarSailOpticalReflectionLaw::evaluateReactionVector(const Eigen::Vector3d& surfaceNormal,
                                                                      const Eigen::Vector3d& incomingDirection) const
    { // Check the signs still though
        double absorptivity;
        double specularReflectivity;
        double diffuseReflectivity;
        double exposedLambertianCoefficient;

        const double cosBetweenNormalAndIncoming = surfaceNormal.dot(-incomingDirection);
        const double signCosBetweenNormalAndIncoming = (cosBetweenNormalAndIncoming >= 0) - (cosBetweenNormalAndIncoming <= 0);

        // Depending on whether the front or the back of the sail is exposed, select appropriate optical coefficients.
        if (cosBetweenNormalAndIncoming > 0){
            absorptivity = frontAbsorptivity_;
            specularReflectivity = frontSpecularReflectivity_;
            diffuseReflectivity = frontDiffuseReflectivity_;
            exposedLambertianCoefficient = frontNonLambertianCoefficient_;
        } else {
            absorptivity = backAbsorptivity_;
            specularReflectivity = backSpecularReflectivity_;
            diffuseReflectivity = backDiffuseReflectivity_;
            exposedLambertianCoefficient = backNonLambertianCoefficient_;
        }


        double EmissionFraction = (backEmissivity_ * backNonLambertianCoefficient_ - frontEmissivity_ * frontNonLambertianCoefficient_)
                /(frontEmissivity_ + backEmissivity_);
        auto reactionFromAbsorptionAndReradiation = absorptivity
                * (incomingDirection + EmissionFraction * surfaceNormal);
        auto reactionFromSpecularReflection = -2 * specularReflectivity
                * cosBetweenNormalAndIncoming * surfaceNormal;
        auto reactionFromDiffuseReflection = diffuseReflectivity
                * (incomingDirection - exposedLambertianCoefficient *  surfaceNormal * signCosBetweenNormalAndIncoming);

        return reactionFromAbsorptionAndReradiation + reactionFromSpecularReflection + reactionFromDiffuseReflection;
    }

Eigen::Matrix3d SolarSailOpticalReflectionLaw::evaluateReactionVectorDerivativeWrtTargetPosition(
            const Eigen::Vector3d& surfaceNormal,
            const Eigen::Vector3d& incomingDirection,
            const double cosineOfAngleBetweenVectors,
            const Eigen::Vector3d& currentReactionVector,
            const Eigen::Matrix3d& sourceUnitVectorPartial,
            const Eigen::Matrix< double, 1, 3 >& cosineAnglePartial )
    {
        throw std::runtime_error(
                "Error. This functionality has not yet been implemented.");
    }

    void SolarSailOpticalReflectionLaw::validateCoefficients() const
    {
        auto sumOfCoeffsFront = frontAbsorptivity_ + frontSpecularReflectivity_ + frontDiffuseReflectivity_;
        auto sumOfCoeffsBack = backAbsorptivity_ + backSpecularReflectivity_ + backDiffuseReflectivity_;
        if (std::fabs(1 - sumOfCoeffsFront) >= 20 * std::numeric_limits<double>::epsilon() ||
        std::fabs(1 - sumOfCoeffsBack) >= 20 * std::numeric_limits<double>::epsilon())
        {
            std::cerr << "Warning, coefficients of optical solar sail reflection law, " <<
                      "should sum to 1" << sumOfCoeffsFront << sumOfCoeffsBack << std::endl;
        }else if (backEmissivity_> 1 || frontEmissivity_> 1){
            std::cerr << "Warning, the emissivity coefficients of the solar sail reflection law, " <<
                      "should have a value between 0 and 1" << std::endl;
        }else if (backNonLambertianCoefficient_ > 1 || frontNonLambertianCoefficient_ > 1){
            std::cerr << "Warning, the Lambertian coefficients of the solar sail reflection law, " <<
                      "should have a value between 0 and 1" << std::endl;
        }
    }
    void SpecularDiffuseMixReflectionLaw::validateCoefficients() const
{
    auto sumOfCoeffs = absorptivity_ + specularReflectivity_ + diffuseReflectivity_;
    if (std::fabs(1 - sumOfCoeffs) >= 20 * std::numeric_limits<double>::epsilon())
    {
        std::cerr << "Warning, coefficients of specular-diffuse-mix reflection law, " <<
                "should sum to 1" << std::endl;
    }
}

Eigen::Vector3d computeMirrorlikeReflection(
        const Eigen::Vector3d& vectorToMirror,
        const Eigen::Vector3d& surfaceNormal)
{
    const double vectorDotNormal = vectorToMirror.dot(surfaceNormal);
    if (vectorDotNormal >= 0)
    {
        // Vector is incident on backside of surface
        return Eigen::Vector3d::Zero();
    }
    else
    {
        return vectorToMirror - 2 * vectorDotNormal * surfaceNormal;
    }
}

} // tudat
} // electromagnetism
