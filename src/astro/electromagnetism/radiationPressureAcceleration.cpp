/*    Copyright (c) 2010-2022, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include "tudat/astro/electromagnetism/radiationPressureAcceleration.h"

#include <functional>

#include <Eigen/Core>
#include <Eigen/Geometry>


namespace tudat
{
namespace electromagnetism
{

void RadiationPressureAcceleration::updateMembers(const double currentTime)
{
    if(currentTime_ != currentTime)
    {
        currentTime_ = currentTime;

        sourceToTargetOccultationModel_->updateMembers(currentTime);

        currentAcceleration_ = calculateAcceleration();
    }
}

Eigen::Vector3d IsotropicPointSourceRadiationPressureAcceleration::calculateAcceleration()
{
    sourceCenterPositionInGlobalFrame_ = sourcePositionFunction_();
    targetCenterPositionInGlobalFrame_ = targetPositionFunction_();
    targetCenterPositionInSourceFrame_ = targetCenterPositionInGlobalFrame_ - sourceCenterPositionInGlobalFrame_;

    // Evaluate irradiances at target position in source frame
    // No rotation to source frame is necessary because isotropic sources are rotation-invariant
    sourceToTargetReceivedFraction = sourceToTargetOccultationModel_->evaluateReceivedFractionFromExtendedSource(
            sourceCenterPositionInGlobalFrame_, sourceBodyShapeModel_, targetCenterPositionInGlobalFrame_ );
    receivedIrradiance =
        sourceModel_->evaluateIrradianceAtPosition( targetCenterPositionInSourceFrame_).front().first * sourceToTargetReceivedFraction;

    if (receivedIrradiance <= 0)
    {
        // Some body is occluding source as seen from target
        return Eigen::Vector3d::Zero();
    }

    if( targetModel_->forceFunctionRequiresLocalFrameInputs( ) )
    {
        Eigen::Quaterniond targetRotationFromLocalToGlobalFrame = targetRotationFromLocalToGlobalFrameFunction_();
        Eigen::Quaterniond targetRotationFromGlobalToLocalFrame = targetRotationFromLocalToGlobalFrame.inverse();

        // Calculate acceleration due to radiation pressure in global frame
        return targetRotationFromLocalToGlobalFrame *
            targetModel_->evaluateRadiationPressureForce(
            receivedIrradiance, targetRotationFromGlobalToLocalFrame * targetCenterPositionInSourceFrame_.normalized() ) /
            targetMassFunction_();
    }
    else
    {
        return targetModel_->evaluateRadiationPressureForce(
                   receivedIrradiance, targetCenterPositionInSourceFrame_.normalized() ) / targetMassFunction_();
    }
}

Eigen::Vector3d PaneledSourceRadiationPressureAcceleration::calculateAcceleration()
{
    // Could use class member to avoid allocation every call, but profiling shows allocation is by far
    // dominated by algebraic operations
    Eigen::Vector3d sourceCenterPositionInGlobalFrame = sourcePositionFunction_(); // position of center of source (e.g. planet)
    Eigen::Quaterniond sourceRotationFromLocalToGlobalFrame = sourceRotationFromLocalToGlobalFrameFunction_();
    Eigen::Quaterniond sourceRotationFromGlobalToLocalFrame = sourceRotationFromLocalToGlobalFrame.inverse();

    Eigen::Vector3d targetCenterPositionInGlobalFrame = targetPositionFunction_();
    Eigen::Quaterniond targetRotationFromLocalToGlobalFrame = targetRotationFromLocalToGlobalFrameFunction_();
    Eigen::Quaterniond targetRotationFromGlobalToLocalFrame = targetRotationFromLocalToGlobalFrame.inverse();

    // Evaluate irradiances from all sub-sources at target position in source frame
    Eigen::Vector3d targetCenterPositionInSourceFrame =
            sourceRotationFromGlobalToLocalFrame * (targetCenterPositionInGlobalFrame - sourceCenterPositionInGlobalFrame);
    auto sourceIrradiancesAndPositions = sourceModel_->evaluateIrradianceAtPosition(targetCenterPositionInSourceFrame);

    // For dependent variables
    double totalReceivedIrradiance = 0;
    unsigned int visibleAndEmittingSourcePanelCounter = 0;

    // Calculate radiation pressure force due to all sub-sources in target frame
    Eigen::Vector3d totalForceInTargetFrame = Eigen::Vector3d::Zero();
    for (auto sourceIrradianceAndPosition : sourceIrradiancesAndPositions) {
        auto sourceIrradiance = std::get<0>(sourceIrradianceAndPosition);
        Eigen::Vector3d sourcePositionInSourceFrame =
                std::get<1>(sourceIrradianceAndPosition); // position of sub-source (e.g. panel)
        Eigen::Vector3d sourcePositionInGlobalFrame =
                sourceCenterPositionInGlobalFrame + sourceRotationFromLocalToGlobalFrame * sourcePositionInSourceFrame;

        auto sourceToTargetReceivedFraction =
                sourceToTargetOccultationModel_->evaluateReceivedFractionFromPointSource(sourcePositionInGlobalFrame,
                                                                                         targetCenterPositionInGlobalFrame);
        auto occultedSourceIrradiance =
                sourceIrradiance * sourceToTargetReceivedFraction;

        if (occultedSourceIrradiance > 0)
        {
            // No body is occluding source as seen from target
            Eigen::Vector3d sourceToTargetDirectionInTargetFrame =
                    targetRotationFromGlobalToLocalFrame * (targetCenterPositionInGlobalFrame - sourcePositionInGlobalFrame).normalized();
            totalForceInTargetFrame +=
                    targetModel_->evaluateRadiationPressureForce(occultedSourceIrradiance, sourceToTargetDirectionInTargetFrame);
            totalReceivedIrradiance += occultedSourceIrradiance;
            visibleAndEmittingSourcePanelCounter += 1;
        }
    }

    // Update dependent variables
    receivedIrradiance = totalReceivedIrradiance;
    visibleAndEmittingSourcePanelCount = visibleAndEmittingSourcePanelCounter;

    // Calculate acceleration due to radiation pressure in global frame
    Eigen::Vector3d acceleration = targetRotationFromLocalToGlobalFrame * totalForceInTargetFrame / targetMassFunction_();
    return acceleration;
}

} // tudat
} // electromagnetism
