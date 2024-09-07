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
    var result: VertexOutput;
    // result.position = vec4(position.xy * time.time, position.zw);
    var xy = position.xy;
    xy *= vec2(cos(time.time), sin(time.time));
    result.position = vec4(xy, position.zw);

    result.color = color;

    return result;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
