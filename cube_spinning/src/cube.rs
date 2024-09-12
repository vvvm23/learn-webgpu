use bytemuck::{Pod, Zeroable};
use std::mem;

use crate::constants;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub color: [f32; 4],
}

fn create_vertex(pos: [f32; 3], color: [f32; 4], offset: f32) -> Vertex {
    let scale = 0.5;
    Vertex {
        pos: [
            (pos[0] + offset) * scale,
            (pos[1] + offset) * scale,
            (pos[2] + offset) * scale,
            1.0,
        ],
        color,
    }
}

fn create_cube_vertices(vertices: &mut Vec<Vertex>, indices: &mut Vec<u16>, offset: f32) {
    let base = vertices.len() as u16;

    #[rustfmt::skip]
    /*
           4 ----7 
         / |   / |
        0 --- 3  |
        |  5--.--6
        | /   | /
        1 --- 2

     */

    #[rustfmt::skip]
    vertices.extend_from_slice(&[
        create_vertex([-0.5, 0.5, -0.5], constants::RED, offset),
        create_vertex([-0.5, -0.5, -0.5], constants::RED, offset),
        create_vertex([0.5, -0.5, -0.5], constants::RED, offset),
        create_vertex([0.5, 0.5, -0.5], constants::RED, offset),

        create_vertex([-0.5, 0.5, 0.5], constants::GREEN, offset),
        create_vertex([-0.5, -0.5, 0.5], constants::GREEN, offset),
        create_vertex([0.5, -0.5, 0.5], constants::GREEN, offset),
        create_vertex([0.5, 0.5, 0.5], constants::GREEN, offset),
    ]);

    #[rustfmt::skip]
    indices.extend([
        // back face
        4, 7, 6,
        6, 5, 4,

        // bottom face
        5, 1, 2,
        2, 6, 5,

        // left face
        0, 4, 5,
        5, 1, 0,

        // front face
        0, 1, 2, 
        2, 3, 0,

        // right face
        3, 2, 6,
        6, 7, 3,

        // top face
        4, 0, 3,
        3, 7, 4,
    ].iter().map(|i| base + *i));
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    create_cube_vertices(&mut vertices, &mut indices, 0.0);

    println!("{:?} vertices, {:?} indices", vertices.len(), indices.len());

    (vertices, indices)
}

pub fn create_buffers(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32, u32) {
    let (vertex_data, index_data) = create_vertices();
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertex_data),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&index_data),
        usage: wgpu::BufferUsages::INDEX,
    });

    let vertex_size = mem::size_of::<Vertex>();
    let num_indices = index_data.len();

    (
        vertex_buffer,
        index_buffer,
        vertex_size as u32,
        num_indices as u32,
    )
}

pub fn create_vertex_buffer_desc(vertex_size: u32) -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: vertex_size as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 4 * 4,
                shader_location: 1,
            },
        ],
    }
}
