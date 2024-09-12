struct VertexOutput {
    @location(0) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

struct TimeUniform {
    time: f32
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> time: TimeUniform;

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(
    @location(0) position: vec4<f32>, 
    @location(1) color: vec4<f32>,
    instance: InstanceInput,
) -> VertexOutput {
    let rotation_mat: mat4x4<f32> = mat4x4<f32>(
        cos(time.time), 0.0, sin(time.time), 0.0,
        0.0, 1.0, 1.0, 0.0,
        -sin(time.time), 0.0, cos(time.time), 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var result: VertexOutput;

    result.position = camera.view_proj * model_matrix * rotation_mat * position;

    result.color = (position + 0.5) / 2.0;

    return result;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
