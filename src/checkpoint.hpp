#ifndef CHECKPOINT_H_
#define CHECKPOINT_H_

// ============================================================
// Pipeline checkpoint system for QuadriFlow
//
// Allows saving/loading full pipeline state at any stage boundary,
// enabling fast benchmarking of individual stages without re-running
// the full pipeline.
//
// Usage:
//   Full run with saves:  ./quadriflow -i model.obj -o out.obj -f 100000 \
//                            -save-at post-field -save-dir /tmp/checkpoints
//   Resume from stage:    ./quadriflow -run-from post-field \
//                            -save-dir /tmp/checkpoints -o out.obj
//   Run single stage:     ./quadriflow -run-from post-field -run-to post-flow \
//                            -save-dir /tmp/checkpoints
//
// Stage names (in pipeline order):
//   post-init         After Initialize()
//   post-orient       After optimize_orientations + ComputeOrientationSingularities
//   post-field        After optimize_positions + ComputePositionSingularities
//   post-edgeinfo     After BuildEdgeInfo + sharp edge setup
//   post-constraints  After BuildIntegerConstraints
//   post-flow         After ComputeMaxFlow
//   post-subdiv1      After first subdivide_edgeDiff
//   pre-ffh           After allow_changes setup, before FixFlipHierarchy
//   post-ffh          After FixFlipHierarchy
//   post-subdiv2      After second subdivide_edgeDiff + FixFlipSat
//   post-extract      After AdvancedExtractQuad + FixValence
//   post-dynamic      After optimize_positions_dynamic (final)
// ============================================================

#include <string>

namespace qflow {

class Parametrizer;

// Stage indices (in pipeline order)
enum PipelineStage {
    STAGE_NONE = -1,
    STAGE_POST_INIT = 0,
    STAGE_POST_ORIENT,
    STAGE_POST_FIELD,
    STAGE_POST_EDGEINFO,
    STAGE_POST_CONSTRAINTS,
    STAGE_POST_FLOW,
    STAGE_POST_SUBDIV1,
    STAGE_PRE_FFH,
    STAGE_POST_FFH,
    STAGE_POST_SUBDIV2,
    STAGE_POST_EXTRACT,
    STAGE_POST_DYNAMIC,
    STAGE_COUNT
};

// Convert stage name string to enum
PipelineStage stage_from_name(const char* name);

// Convert enum to display name
const char* stage_name(PipelineStage s);

// Checkpoint file header (stored at start of each checkpoint file)
struct CheckpointHeader {
    char magic[4];          // "QFC\0"
    int version;            // format version (1)
    char stage[64];         // stage name
    int fixflip_strategy;   // -ff flag value
    int dse_strategy;       // -dse flag value
    int subdiv_strategy;    // -subdiv flag value
    int flow_strategy;      // -flow flag value
    int flag_preserve_sharp;
    int flag_preserve_boundary;
    int flag_adaptive_scale;
    int flag_aggresive_sat;
    int flag_minimum_cost_flow;
    int target_faces;       // -f flag value
    char input_mesh[256];   // input mesh path
    long long timestamp;    // unix timestamp
    char reserved[184];     // padding for future use (was 192, used 8 for new fields)
};

// Save full Parametrizer state at a given stage
void save_checkpoint(const Parametrizer& p, PipelineStage stage,
                     const char* dir, const char* input_mesh, int target_faces);

// Load checkpoint, returns stage that was saved
PipelineStage load_checkpoint(Parametrizer& p, const char* dir, PipelineStage stage);

// Check if a checkpoint file exists for a given stage
bool checkpoint_exists(const char* dir, PipelineStage stage);

} // namespace qflow

#endif
