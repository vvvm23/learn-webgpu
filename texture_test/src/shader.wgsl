struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec4<f32>, 
) -> VertexOutput {
    var out: VertexOutput;
    out.position = position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(1.0, 1.0, 1.0, 1.0);
}
