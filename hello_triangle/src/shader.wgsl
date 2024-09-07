struct VertexOutput {
    @location(0) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec4<f32>, 
    @location(1) color: vec4<f32>
) -> VertexOutput {
    // let x = f32(i32(in_vertex_index) - 1);
    // let y = f32(i32(in_vertex_index & 1u) * 2 - 1);
    // return vec4<f32>(x, y, 0.0, 1.0);
    var result: VertexOutput;
    result.color = color;
    result.position = position;

    return result;
}

@group(0)
@binding(1)
var<uniform> color: vec4<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
