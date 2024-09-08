struct VertexOutput {
    @location(0) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

struct TimeUniform {
    time: f32
}

@group(0) @binding(0)
var<uniform> time: TimeUniform;

@vertex
fn vs_main(
    @location(0) position: vec4<f32>, 
    @location(1) color: vec4<f32>
) -> VertexOutput {
    let ROTATE: mat4x4<f32> = mat4x4<f32>(
        1.0, 0.0, 0.0, 0.0,
        0.0, cos(time.time), sin(time.time), 0.0,
        0.0, -sin(time.time), cos(time.time), 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    var result: VertexOutput;

    result.position = ROTATE * position;
    result.color = color;

    return result;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
