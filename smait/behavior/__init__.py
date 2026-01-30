"""
SMAIT HRI System v2.0 - Behavior Tree Module

Replaces the FSM with py_trees for composable, parallel behaviors.
Enables advanced features like backchanneling and VAP.
"""

from smait.behavior.tree import HRIBehaviorTree, create_hri_tree
from smait.behavior.behaviors import (
    ListenForSpeech,
    TrackFaces,
    VerifySpeaker,
    GenerateResponse,
    Backchannel,
    CheckTimeout,
)

__all__ = [
    'HRIBehaviorTree',
    'create_hri_tree',
    'ListenForSpeech',
    'TrackFaces',
    'VerifySpeaker',
    'GenerateResponse',
    'Backchannel',
    'CheckTimeout',
]
