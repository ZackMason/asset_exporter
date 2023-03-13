#include "core.hpp"
#include <fstream>
#include <filesystem>
#include <sstream>
#include <algorithm>

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

#include "physics.hpp"

#include "nlohmann/json.hpp"

using json = nlohmann::json;

/////////////////////////////////////////////
// Magic Numbers

namespace magic {
constexpr u64 meta = 0xfeedbeeff04edead;
constexpr u64 vers = 0x2;
constexpr u64 mesh = 0x1212121212121212;
constexpr u64 text = 0x1212121212121213;
constexpr u64 physics = 0x1212121212121214;

constexpr u64 table_start = 0x7abe17abe1;
};

std::string get_extension(const std::string& str) {
    return str.substr(str.find_last_of('.') + 1);
}

bool has_extension(const std::string& str, const std::string& ext) {
    return get_extension(str) == ext;
}

enum struct PhysicsColliderType {
    TRIMESH,
    CONVEX,
    HEIGHTMAP,
    SIZE
};

struct physics_config_files_t {
    std::vector<std::string> trimeshes;
    std::vector<std::string> convex;
};

inline physics_config_files_t
load_physics_config(std::string_view path) {

}

using resource_t = std::vector<u8>;

struct resource_table_entry_t {
    std::string name;
    u64 type{0};
    u64 size{0};
};

struct pack_file_t {
    u64 meta{};
    u64 vers{};

    u64 file_count{0};
    u64 resource_size{0};

    u64 table_start{magic::table_start};

    std::vector<resource_table_entry_t> table;

    std::vector<resource_t> resources;
};

void serialize(utl::serializer_t& s, const resource_table_entry_t& t) {
    s.serialize(t.name);
    s.serialize(t.type);
    s.serialize(t.size);
}

void serialize(utl::serializer_t& s, pack_file_t& f) {
    s.serialize(f.meta);
    s.serialize(f.vers);
    
    s.serialize(f.file_count);
    s.serialize(f.resource_size);

    s.serialize(f.table_start);

    s.serialize<resource_table_entry_t>(f.table, serialize);

    s.serialize(f.resources);
}


void
close_pack_file(
    pack_file_t* file_pack,
    std::string_view out_path
) {

    file_pack->meta = magic::meta;
    file_pack->vers = magic::vers;

    std::vector<u8> data;
    utl::serializer_t serializer{data};

    serialize(serializer, *file_pack);
    
    std::ofstream file{fmt_str("{}", out_path), std::ios::binary};
    file.write((const char*)data.data(), data.size());
    file.close();
    
    gen_info("packer", "Packed {} files", file_pack->file_count);

}

void
pack_file(
    pack_file_t* file_pack, 
    std::string_view path, 
    std::vector<u8>&& data,
    u64 file_type
) {
    gen_info("export", "Writing file: {}, offset: {}, size: {}", path, file_pack->resource_size, data.size());

    file_pack->table.emplace_back(resource_table_entry_t(std::string{path}, file_type, data.size()));

    file_pack->resource_size += data.size();

    file_pack->resources.emplace_back(std::move(data));
    
    file_pack->file_count++;
}

void serialize(utl::serializer_t& s, const gfx::vertex_t& v) {
    s.serialize(v.pos.x);
    s.serialize(v.pos.y);
    s.serialize(v.pos.z);
    
    s.serialize(v.nrm.x);
    s.serialize(v.nrm.y);
    s.serialize(v.nrm.z);
    
    s.serialize(v.col.x);
    s.serialize(v.col.y);
    s.serialize(v.col.z);
    
    s.serialize(v.tex.x);
    s.serialize(v.tex.y);
}

template <typename Vertex>
struct mesh_t {
    std::string mesh_name;

    std::vector<Vertex> vertices{};
    std::vector<u32>    indices{};
};

template<typename Vertex>
using loaded_mesh_t = std::vector<mesh_t<Vertex>>;

template<typename Vertex>
void process_mesh(
    aiMesh* mesh, 
    const aiScene* scene, 
    loaded_mesh_t<Vertex>& results
) {
    results.back().mesh_name = (mesh->mName.C_Str());
    
    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        
        vertex.pos = utl::make_v3f(mesh->mVertices[i]);

        if (mesh->mNormals) {
            vertex.nrm = glm::normalize(utl::make_v3f(mesh->mNormals[i]));
        }
              
        if (mesh->mTextureCoords[0]) {
            glm::vec2 vec;
            vec.x = mesh->mTextureCoords[0][i].x;
            vec.y = mesh->mTextureCoords[0][i].y;
            vertex.tex = vec;
        } else {
            vertex.tex = glm::vec2(0.0f, 0.0f);
        }

        results.back().vertices.push_back(vertex);
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        assert(face.mNumIndices == 3);
        for (unsigned int j = 0; j < face.mNumIndices; j++)
            results.back().indices.push_back(face.mIndices[j]);
    }
    // aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
    // aiColor3D color;
    // material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
    // results.back().material.base_color = { color.r, color.g, color.b };

    // material->Get(AI_MATKEY_COLOR_AMBIENT, color);
    // results.back().material.emissive = { color.r };    

    // if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
    //     aiString diffuse_name;
    //     material->GetTexture(aiTextureType_DIFFUSE, 0, &diffuse_name);
    //     results.back().material.albedo_texture = save_string(diffuse_name.C_Str());
    // }

    // aiString name;
    // material->Get(AI_MATKEY_NAME, name);
    // results.back().material_name = save_string(name.C_Str());
}

template<typename Vertex>
void process_node(
    aiNode *start,
    const aiScene *scene, 
    loaded_mesh_t<Vertex>& results
) {
    std::stack<const aiNode*> stack;
    stack.push(scene->mRootNode);
    while (!stack.empty()) {
        const aiNode* node = stack.top();
        stack.pop();

        for(unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            results.emplace_back();
            process_mesh(mesh, scene, results);
        }

        for(unsigned int i = 0; i < node->mNumChildren; i++) {
            stack.push(node->mChildren[i]);
        }
    }
}

template <typename Vertex>
inline void
export_physics_mesh(
    const loaded_mesh_t<Vertex>& meshes,
    std::vector<u8>& physics_data,
    PhysicsColliderType physics_type,
    phys::physx_state_t& state
) {
    std::vector<v3f> positions;
    std::vector<u32> indices;

    for (const auto& mesh: meshes) {
        std::transform(mesh.vertices.cbegin(), 
            mesh.vertices.cend(), 
            std::back_inserter(positions), 
            [](const auto& vertex) {
                return vertex.pos;
        });
        std::transform(mesh.indices.cbegin(), 
            mesh.indices.cend(), 
            std::back_inserter(indices), 
            [](const auto id) {
                return id;
        });
    }

    physx::PxDefaultMemoryOutputStream buf;

    switch(physics_type) {
        case PhysicsColliderType::CONVEX: {
            physx::PxConvexMeshDesc convexDesc;
            convexDesc.points.count     = (physx::PxU32)positions.size();
            convexDesc.points.stride    = sizeof(v3f);
            convexDesc.points.data      = positions.data();
            //convexDesc.polygons.count   = indices ? static_cast<physx::PxU32>(indices->size()) : 0;
            //convexDesc.polygons.stride  = indices ? sizeof(unsigned int) : 0;
            //convexDesc.polygons.data    = indices ? indices->data() : nullptr;
            convexDesc.flags            = physx::PxConvexFlag::eCOMPUTE_CONVEX;

            if(!state.cooking->cookConvexMesh(convexDesc, buf)) {
                gen_error("physx", "Error cooking convex");
            }
        } break;
        case PhysicsColliderType::TRIMESH: {
            
            physx::PxTriangleMeshDesc meshDesc;
            meshDesc.points.data  = positions.data();
            meshDesc.points.count = (physx::PxU32)positions.size();
            meshDesc.points.stride = sizeof(v3f);
            meshDesc.triangles.data = indices.data();
            meshDesc.triangles.count = (physx::PxU32)indices.size()/3;
            meshDesc.triangles.stride = sizeof(u32) * 3;

            if (!state.cooking->cookTriangleMesh(meshDesc, buf)) {
                gen_error("physx", "Error cooking trimesh");
            }
        } break;
    }

    physics_data.reserve(buf.getSize());
    physics_data.resize(buf.getSize());

    std::memcpy(physics_data.data(), buf.getData(), buf.getSize());
}

template <typename Vertex>
inline std::vector<u8>
export_mesh(
    arena_t* arena,
    std::string_view path,
    std::vector<u8>* physics_data = 0,
    PhysicsColliderType physics_type = PhysicsColliderType::CONVEX,
    phys::physx_state_t* physx_state = 0
) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(fmt::format("{}", path), 
        aiProcess_Triangulate | 
        aiProcess_JoinIdenticalVertices | 
        aiProcess_GenNormals | 
        aiProcess_GlobalScale
    );

    if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error(importer.GetErrorString());
    }

    loaded_mesh_t<Vertex> results;
    process_node(scene->mRootNode, scene, results);

    std::vector<u8> data;
    utl::serializer_t serializer{data};
    
    serializer.serialize(magic::meta);
    serializer.serialize(magic::vers);
    serializer.serialize(magic::mesh);

    serializer.serialize(results.size()); // number of meshes
    for (auto& mesh: results) {
        serializer.serialize(mesh.mesh_name);
        serializer.serialize(mesh.vertices.size());
        for (const auto& vertex: mesh.vertices) {
            serialize(serializer, vertex);
        }
        serializer.serialize(std::span{mesh.indices});
    }
    gen_info("mesh", "Loaded {} meshes", results.size());

    if (physics_data) {
        export_physics_mesh(results, *physics_data, physics_type, *physx_state);
    }

    return data;
}

void pack_asset_directory(
    std::string_view dir,
    std::string_view out_path
) {
    phys::physx_state_t physx_state;
    phys::init_physx_state(physx_state);

    pack_file_t* packed_file = new pack_file_t;
    constexpr size_t arena_size = megabytes(512);
    arena_t* arena = new arena_t;
    arena->start = new u8[arena_size];
    arena->size = arena_size;

    std::ifstream physx_defs{fmt_str("{}/physx_defs.json", dir)};
    physics_config_files_t physx_config;

    if (physx_defs.is_open()) {
        json j;
        physx_defs >> j;
        
        for (const auto mesh: j["trimesh"]) {
            gen_info("phyx::config", "Trimesh: {}", mesh);
            physx_config.trimeshes.push_back(mesh);
        }
        for (const auto mesh: j["convex"]) {
            gen_info("phyx::config", "Convex: {}", mesh);
            physx_config.convex.push_back(mesh);
        }
    }

    for (const auto& entry: std::filesystem::recursive_directory_iterator(fmt_str("{}", dir))) {
        std::string file_name = entry.path().string();
        std::replace(file_name.begin(), file_name.end(), '\\', '/');
        if (has_extension(file_name, "obj") || has_extension(file_name, "fbx")) {

            bool make_physics = false;
            PhysicsColliderType collider = PhysicsColliderType::CONVEX;

            if (std::find(physx_config.convex.begin(), physx_config.convex.end(), file_name) != physx_config.convex.end()) {
                gen_warn("export", "convex");
                make_physics = true;
            }
            if (std::find(physx_config.trimeshes.begin(), physx_config.trimeshes.end(), file_name) != physx_config.trimeshes.end()) {
                gen_warn("export", "trimesh");
                collider = PhysicsColliderType::TRIMESH;
                make_physics = true;
            }

            std::vector<u8> physics_data;
            std::vector<u8> data = export_mesh<gfx::vertex_t>(
                arena, file_name, 
                make_physics ? &physics_data : 0,
                collider,
                make_physics ? &physx_state : 0
            );
            pack_file(packed_file, file_name, std::move(data), magic::mesh);

            if (make_physics) {
                if (collider == PhysicsColliderType::TRIMESH) {
                    pack_file(packed_file, file_name+".trimesh.physx", std::move(physics_data), magic::physics);
                } else if (collider == PhysicsColliderType::CONVEX) {
                    pack_file(packed_file, file_name+".convex.physx", std::move(physics_data), magic::physics);
                }
            }

        } else if (
            has_extension(file_name, "txt") || 
            has_extension(file_name, "json")
        ) {
            std::vector<u8> data;
            utl::serializer_t saver{data};
            std::ifstream file{file_name};
            saver.serialize((std::stringstream{} << file.rdbuf()).str());
            pack_file(packed_file, file_name, std::move(data), magic::text);
            
        } else {
            gen_warn("export", "Unknown file type: {}", file_name);
        }
    }

    close_pack_file(packed_file, out_path);
}

int main(int argc, const char* argv[]) {
    if (argc >= 4) {
        
        constexpr size_t arena_size = megabytes(512);
        arena_t* arena = new arena_t;
        arena->start = new u8[arena_size];
        arena->size = arena_size;

        std::string_view flag = argv[1];
        std::string_view param = argv[2];
        std::string_view output = argv[3];

        if (flag == "mesh") {
            auto data = export_mesh<gfx::vertex_t>(arena, param);

            std::ofstream file{fmt_str("{}", output), std::ios::binary};
            file.write((const char*)data.data(), data.size());
        } else if (flag == "dir") {
            pack_asset_directory(param, output);
        }
    }

    return 0;
}