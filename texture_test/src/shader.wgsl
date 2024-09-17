struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>
};

struct TimeUniform {
    time: f32
}

@group(1) @binding(0)
var<uniform> time: TimeUniform;

@vertex
fn vs_main(
    @location(0) position: vec4<f32>, 
    @location(1) tex_coords: vec2<f32>,
) -> VertexOutput {
    let rotation_mat: mat4x4<f32> = mat4x4<f32>(
        cos(time.time), -sin(time.time), 0.0, 0.0,
        sin(time.time), cos(time.time), 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    var out: VertexOutput;
    out.position = rotation_mat * position;
    out.tex_coords = tex_coords;
    return out;
}


@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // return vec4(1.0, 1.0, 1.0, 1.0);
    return textureSample(t_diffuse, s_diffuse, in.tex_coords + time.time);
}
