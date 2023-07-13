#define _SILENCE_CXX20_CISO646_REMOVED_WARNING
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

constexpr auto make_magic(const char key[8]) {
    u64 res=0;
    for(size_t i=0;i<4;i++) { res = (res<<8) | key[i];}
    return res;
}

namespace magic {
constexpr u64 meta = 0xfeedbeeff04edead;
constexpr u64 vers = 0x2;
constexpr u64 mesh = 0x1212121212121212;
constexpr u64 text = 0x1212121212121213;
constexpr u64 skel = 0x1212691212121241;
constexpr u64 anim = 0x1212691212121269;
constexpr u64 physics = 0x1212121212121214;
constexpr u64 mate = make_magic("MATERIAL");

constexpr u64 table_start = 0x7abe17abe1;

};

namespace utl::anim {
    bone_timeline_t::bone_timeline_t(std::string_view pname, bone_id_t pID, const aiNodeAnim* channel)
        : id(pID), transform(1.0f)
    {
        std::memcpy(name, pname.data(), pname.size());
        assert(channel->mNumPositionKeys < 512);
        assert(channel->mNumRotationKeys < 512);
        assert(channel->mNumScalingKeys < 512);
        for (size_t positionIndex = 0; positionIndex < channel->mNumPositionKeys; ++positionIndex)
        {
            aiVector3D pos = channel->mPositionKeys[positionIndex].mValue;
            float timeStamp = (float)channel->mPositionKeys[positionIndex].mTime;
            keyframe<v3f>& data = positions[position_count++];
            data.value = v3f(pos.x, pos.y, pos.z);
            data.time = timeStamp;
        }

        for (size_t rotationIndex = 0; rotationIndex < channel->mNumRotationKeys; ++rotationIndex)
        {
            aiQuaternion orientation = channel->mRotationKeys[rotationIndex].mValue;
            float timeStamp = (float)channel->mRotationKeys[rotationIndex].mTime;
            keyframe<glm::quat>& data = rotations[rotation_count++];
            data.value = glm::quat(orientation.w, orientation.x, orientation.y, orientation.z);
            data.time = timeStamp;
        }

        for (size_t keyIndex = 0; keyIndex < channel->mNumScalingKeys; ++keyIndex)
        {
            aiVector3D scale = channel->mScalingKeys[keyIndex].mValue;
            float timeStamp = (float)channel->mScalingKeys[keyIndex].mTime;
            keyframe<v3f>& data = scales[scale_count++];
            data.value = v3f(scale.x, scale.y, scale.z);
            data.time = timeStamp;
        }
    }

    void skeleton_t::load(const std::string& path)
    {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(fmt::format("{}", path), 
            aiProcess_Triangulate | 
            aiProcess_JoinIdenticalVertices | 
            aiProcess_GenNormals | 
            aiProcess_GlobalScale
        );

        if (!scene) {
            gen_error(__FUNCTION__, "{}", importer.GetErrorString());
        }

        assert(scene && scene->mRootNode);

        struct bone_pair_t {
            std::string name;
            m44 offset;
        };

        const auto convert_matrix = [](const aiMatrix4x4& from)	{
            glm::mat4 to;
            to[0][0] = from.a1; to[1][0] = from.a2; to[2][0] = from.a3; to[3][0] = from.a4;
            to[0][1] = from.b1; to[1][1] = from.b2; to[2][1] = from.b3; to[3][1] = from.b4;
            to[0][2] = from.c1; to[1][2] = from.c2; to[2][2] = from.c3; to[3][2] = from.c4;
            to[0][3] = from.d1; to[1][3] = from.d2; to[2][3] = from.d3; to[3][3] = from.d4;
            return to;
        };


        const auto parse_bone_names = [=](const aiScene* scene) -> std::vector<bone_pair_t> {
            assert(scene);
            assert(scene->HasMeshes());
            assert(scene->mMeshes);

            std::vector<bone_pair_t> bone_names;

            for (size_t i{ 0 }; i < scene->mNumMeshes; i++) {
                const auto mesh = scene->mMeshes[i];
                if (mesh->HasBones()) {
                    for (size_t j{ 0 }; j < mesh->mNumBones; j++) {
                        std::string bone_name = mesh->mBones[j]->mName.C_Str();
                        if (std::find_if(bone_names.begin(), bone_names.end(), [&](auto& a){ return a.name == bone_name; }) == bone_names.end()) {
                            bone_names.push_back({
                                bone_name, 
                                convert_matrix(mesh->mBones[j]->mOffsetMatrix)
                            });
                        }
                    }
                }
            }
            gen_info(__FUNCTION__, "Found {} bones", bone_names.size());
            return bone_names;
        };

        const auto read_heirarchy_data = [=](auto& skeleton, const aiNode* root, std::vector<bone_pair_t> bone_names) {
            struct parse_node_t {
                const aiNode* node{nullptr};
                bone_id_t parent{-1};
            };
            std::stack<parse_node_t> stack;
            stack.push({root, -1});

            while (!stack.empty()) {
                parse_node_t node = std::move(stack.top());
                stack.pop();

                assert(node.node);    

                bone_id_t bone_id = -1;
                //if (std::find(bone_names.cbegin(), bone_names.cend(), node.node->mName.C_Str()) != bone_names.cend()) {

                    skeleton_bone_t bone;
                    bone.name_hash = sid(std::string_view{node.node->mName.C_Str(), node.node->mName.length});
                    bone.parent = node.parent;
                    for (auto& [bone_name, offset]: bone_names) {
                        if (bone_name == node.node->mName.C_Str()) {
                            bone.offset = offset;
                        }
                    }
            
                    assert(skeleton.bone_count < skeleton_t::max_bones_());
                    bone_id = (i32)skeleton.bone_count++;
                    skeleton.bones[bone_id] = bone;
                //}

                for (size_t i = 0; i < node.node->mNumChildren; i++) {
                    stack.push({node.node->mChildren[i], bone_id});
                }
            }
        };
        auto bone_names = parse_bone_names(scene);
        read_heirarchy_data(*this, scene->mRootNode, bone_names);
    }
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
    
    gen_warn("packer", "Packed {} files", file_pack->file_count);

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

void serialize(utl::serializer_t& s, const gfx::skinned_vertex_t& v) {
    s.serialize(v.pos.x);
    s.serialize(v.pos.y);
    s.serialize(v.pos.z);
    
    s.serialize(v.nrm.x);
    s.serialize(v.nrm.y);
    s.serialize(v.nrm.z);

    s.serialize(v.tex.x);
    s.serialize(v.tex.y);

    s.serialize(v.bone_id[0]);
    s.serialize(v.bone_id[1]);
    s.serialize(v.bone_id[2]);
    s.serialize(v.bone_id[3]);

    s.serialize(v.weight.x);
    s.serialize(v.weight.y);
    s.serialize(v.weight.z);
    s.serialize(v.weight.w);
}

struct material_info_t {
    char name[64]{};
    v4f color;
    f32 roughness{0.5f};
    f32 metallic{0.0f};
    f32 emission{0.0f};

    u64 albedo_id;
    u64 normal_id;
    char albedo[128];
    char normal[128];
};

template <typename Vertex>
struct mesh_t {
    std::string mesh_name;

    std::vector<Vertex> vertices{};
    std::vector<u32>    indices{};
    material_info_t     material{};
};

void set_vertex_bone_data(gfx::skinned_vertex_t& vertex, int bone_id, float weight) {
    for (int i = 0; i < 4; ++i) {
        if (vertex.bone_id[i] < 0) {
            vertex.weight[i] = weight;
            vertex.bone_id[i] = bone_id;
            break;
        }
    }
}

void extract_bone_weight(
    utl::anim::skeleton_t& skeleton,
    std::vector<gfx::skinned_vertex_t>& vertices, 
    aiMesh* mesh, 
    const aiScene* scene
) {
    for (size_t boneIndex = 0; boneIndex < mesh->mNumBones; ++boneIndex) {
        const std::string bone_name = mesh->mBones[boneIndex]->mName.C_Str();
        const int bone_id = skeleton.find_bone_id(sid(bone_name));
        
        if (bone_id == -1) {
            gen_warn(__FUNCTION__, "Skeleton is missing bone: {}", bone_name);
        }
        assert(bone_id != -1 && "Skeleton is missing bones");

        auto weights = mesh->mBones[boneIndex]->mWeights;
        int numWeights = mesh->mBones[boneIndex]->mNumWeights;

        for (int weightIndex = 0; weightIndex < numWeights; ++weightIndex) {
            int vertexId = weights[weightIndex].mVertexId;
            float weight = weights[weightIndex].mWeight;
            assert(vertexId < vertices.size());
            set_vertex_bone_data(vertices[vertexId], bone_id, weight);
        }
    }
}

template<typename Vertex>
using loaded_mesh_t = std::vector<mesh_t<Vertex>>;

template<typename Vertex>
void process_mesh(
    aiMesh* mesh, 
    const aiScene* scene, 
    loaded_mesh_t<Vertex>& results,
    utl::anim::skeleton_t* skeleton = nullptr
) {
    std::vector<Vertex> vertices;
    auto& r = results.back();
    r.mesh_name = (mesh->mName.C_Str());
    
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

        if constexpr (std::is_same_v<Vertex, gfx::skinned_vertex_t>) {
            for (u32 j = 0; j < 4; j++) {
                vertex.bone_id[j] = std::numeric_limits<u32>::max();
                vertex.weight[j] = 0.0f;
            }
        }

        vertices.push_back(vertex);
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        // assert(face.mNumIndices == 3);
        for (unsigned int j = 0; j < face.mNumIndices; j++)
            r.indices.push_back(face.mIndices[j]);
    }

    if constexpr (std::is_same_v<Vertex, gfx::skinned_vertex_t>) {
        if (skeleton) {
            extract_bone_weight(*skeleton, vertices, mesh, scene);
        } else {
            gen_warn(__FUNCTION__, "No skeleton");
        }

        std::transform(
            r.indices.begin(), 
            r.indices.end(), 
            std::back_inserter(r.vertices),
            [&](auto i){ return vertices[i]; });

        r.indices.clear(); 
    } else {
        r.vertices = std::move(vertices);
    }

    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
    aiColor3D color;
    material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
    r.material.color = { color.r, color.g, color.b, 1.0f};

    material->Get(AI_MATKEY_COLOR_AMBIENT, color);
    r.material.emission = { color.r };    

    material->Get(AI_MATKEY_METALLIC_FACTOR, r.material.metallic);
    material->Get(AI_MATKEY_ROUGHNESS_FACTOR, r.material.roughness);

    if (material->GetTextureCount(aiTextureType_BASE_COLOR) > 0) {
        aiString diffuse_name;
        material->GetTexture(aiTextureType_BASE_COLOR, 0, &diffuse_name);
        if (diffuse_name.data[0] == '*') {
            diffuse_name = scene->GetEmbeddedTexture(diffuse_name.C_Str())->mFilename;
        }
        if (diffuse_name.length > array_count(r.material.albedo)) {
            gen_warn(__FUNCTION__, "Albedo texture path is too large", diffuse_name.C_Str());
        } else {
            gen_warn(__FUNCTION__, "Albedo Texture: {}", diffuse_name.data);
        }
        std::memcpy(r.material.albedo, diffuse_name.C_Str(), diffuse_name.length);
    }
    if (material->GetTextureCount(aiTextureType_NORMAL_CAMERA) > 0) {
        aiString normal_name;
        material->GetTexture(aiTextureType_NORMAL_CAMERA, 0, &normal_name);
        if (normal_name.data[0] == '*') {
            normal_name = scene->GetEmbeddedTexture(normal_name.C_Str())->mFilename;
        }
        if (normal_name.length > array_count(r.material.albedo)) {
            gen_warn(__FUNCTION__, "Normal texture path is too large", normal_name.C_Str());
        } else {
            gen_warn(__FUNCTION__, "Normal Texture: {}", normal_name.data);
        }
        std::memcpy(r.material.normal, normal_name.C_Str(), normal_name.length);
    }

    aiString name;
    material->Get(AI_MATKEY_NAME, name);
    
    if (name.length > array_count(r.material.name)) {
        gen_warn(__FUNCTION__, "Albedo texture path is too larger", name.C_Str());
    }
    std::memcpy(r.material.name, name.C_Str(), name.length);
}

void load_animation(
    utl::anim::animation_t& animation,
    const aiNode* ai_root,
    const aiAnimation* ai_anim, 
    const utl::anim::skeleton_t& skeleton
) {
    assert(ai_anim);
    assert(ai_root);
    const auto size = ai_anim->mNumChannels;
    std::memcpy(animation.name, ai_anim->mName.C_Str(), std::min((u32)array_count(animation.name), ai_anim->mName.length));

    const auto convert_matrix = [](const aiMatrix4x4& from)	{
        glm::mat4 to;
        to[0][0] = from.a1; to[1][0] = from.a2; to[2][0] = from.a3; to[3][0] = from.a4;
        to[0][1] = from.b1; to[1][1] = from.b2; to[2][1] = from.b3; to[3][1] = from.b4;
        to[0][2] = from.c1; to[1][2] = from.c2; to[2][2] = from.c3; to[3][2] = from.c4;
        to[0][3] = from.d1; to[1][3] = from.d2; to[2][3] = from.d3; to[3][3] = from.d4;
        return to;
    };


    for(size_t i{0}; i < size; i++) {
        const auto channel = ai_anim->mChannels[i];

        const std::string bone_name = channel->mNodeName.C_Str();
        const auto bone_id = skeleton.find_bone_id(sid(bone_name));
        if (bone_id == -1) {
            gen_error(__FUNCTION__, "Failed to find bone {}", bone_name);
            continue;
        }
        assert(bone_id != -1);
        const auto& bone = skeleton.find_bone(sid(bone_name));

        animation.node_count = bone_id >= animation.node_count ? bone_id + 1 : animation.node_count;
        animation.nodes[bone_id].offset = bone.offset;
        animation.nodes[bone_id].parent = bone.parent;
        const auto ai_node = ai_root->FindNode(bone_name.c_str());
        if (ai_node) {
            animation.nodes[bone_id].transform = convert_matrix(ai_node->mTransformation);
        }
        else {
            gen_warn(__FUNCTION__, "Failed to find node for transform: {}", bone_name);
            animation.nodes[bone_id].transform = m44(1.0f);
        }
        animation.nodes[bone_id].bone.emplace(std::string_view{channel->mNodeName.data, channel->mNodeName.length}, bone_id, channel);
    }
}

bool process_animations(
    aiNode* start,
    const aiScene* scene,
    std::vector<utl::anim::animation_t>& results,
    const utl::anim::skeleton_t& skeleton
) {
    for (size_t i{ 0 }; i < scene->mNumAnimations; i++) {
        auto* ai_animation = scene->mAnimations[i];
        results.push_back(utl::anim::animation_t{});
        auto& animation = results.back();
        animation.duration = (f32)ai_animation->mDuration;
        animation.ticks_per_second = (i32)ai_animation->mTicksPerSecond;
        load_animation(animation, scene->mRootNode, ai_animation, skeleton);
    }

    return true;
}

template<typename Vertex>
bool process_node(
    aiNode *start,
    const aiScene *scene, 
    loaded_mesh_t<Vertex>& results,
    auto&& process, u64 looking_for, utl::anim::skeleton_t* skeleton = 0
) {
    if (looking_for == magic::skel && !skeleton) {
        return false;
    }
    bool has_looking_for = looking_for == magic::mesh;
    std::stack<const aiNode*> stack;
    stack.push(scene->mRootNode);
    while (!stack.empty()) {
        const aiNode* node = stack.top();
        stack.pop();

        for(unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

            if (looking_for == magic::skel && skeleton) {
                if (mesh->HasBones()) {
                    gen_info(__FUNCTION__, "Mesh {} has {} bones", mesh->mName.C_Str(), mesh->mNumBones);
                    has_looking_for = true;
                } else {
                    gen_info(__FUNCTION__, "Mesh {} has no bones", mesh->mName.C_Str());
                }
            }

            results.emplace_back();
            process(mesh, scene, results, skeleton);
        }

        for(unsigned int i = 0; i < node->mNumChildren; i++) {
            stack.push(node->mChildren[i]);
        }
    }

    return has_looking_for;
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
        if (physics_type==PhysicsColliderType::TRIMESH) {
            if (mesh.mesh_name.find("<trimesh>") == std::string::npos)
            {
                gen_warn(__FUNCTION__, "Skipping submesh: {}", mesh.mesh_name);
                continue;
            } else {
                gen_warn(__FUNCTION__, "Cooking submesh: {}", mesh.mesh_name);
            }
        }
        if (physics_type==PhysicsColliderType::CONVEX) {
            if (mesh.mesh_name.find("<convex>") == std::string::npos)
            {
                gen_warn(__FUNCTION__, "Skipping submesh: {}", mesh.mesh_name);
                continue;
            } else {
                gen_warn(__FUNCTION__, "Cooking submesh: {}", mesh.mesh_name);
            }
        }

        u32 offset = safe_truncate_u64(positions.size());
        std::transform(mesh.vertices.cbegin(), 
            mesh.vertices.cend(), 
            std::back_inserter(positions), 
            [](const auto& vertex) {
                return vertex.pos;
        });
        std::transform(mesh.indices.cbegin(), 
            mesh.indices.cend(), 
            std::back_inserter(indices), 
            [=](const auto id) {
                return offset + id;
        });
    }

    if (positions.size() == 0) return;

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
inline [[nodiscard]] std::vector<u8>
export_mesh(
    arena_t* arena,
    std::string_view path,
    std::vector<u8>* physics_data = 0,
    PhysicsColliderType physics_type = PhysicsColliderType::CONVEX,
    phys::physx_state_t* physx_state = 0
) {
    return {};
}

template <>
inline [[nodiscard]] std::vector<u8>
export_mesh<gfx::vertex_t>(
    arena_t* arena,
    std::string_view path,
    std::vector<u8>* physics_data,
    PhysicsColliderType physics_type,
    phys::physx_state_t* physx_state
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

    loaded_mesh_t<gfx::vertex_t> results;
    process_node<gfx::vertex_t>(scene->mRootNode, scene, results, process_mesh<gfx::vertex_t>, magic::mesh);

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

    serializer.serialize(magic::mate);

    for (auto& mesh: results) {
        serializer.serialize_bytes(mesh.material);
    }

    gen_info("mesh", "Loaded {} meshes", results.size());

    if (physics_data) {
        export_physics_mesh(results, *physics_data, physics_type, *physx_state);
    }

    return data;
}

template <>
inline [[nodiscard]] std::vector<u8>
export_mesh<gfx::skinned_vertex_t>(
    arena_t* arena,
    std::string_view path,
    std::vector<u8>* physics_data,
    PhysicsColliderType physics_type,
    phys::physx_state_t* physx_state
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

    utl::anim::skeleton_t* skeleton_ = new utl::anim::skeleton_t;
    utl::anim::skeleton_t& skeleton = *skeleton_;
    skeleton.load(std::string{path});
    if (skeleton.bone_count == 0) {
        gen_warn(__FUNCTION__, "Skeleton didnt load any boners");
        return {};
    }
    std::vector<utl::anim::animation_t> animations{};
    process_animations(scene->mRootNode, scene, animations, skeleton);

    loaded_mesh_t<gfx::skinned_vertex_t> results;
    if (process_node<gfx::skinned_vertex_t>(scene->mRootNode, scene, results, process_mesh<gfx::skinned_vertex_t>, magic::skel, &skeleton) == false) {
        return {};
    }
    
    std::vector<u8> data;
    utl::serializer_t serializer{data};
    
    serializer.serialize(magic::meta);
    serializer.serialize(magic::vers);
    serializer.serialize(magic::skel);

    serializer.serialize(results.size()); // number of meshes
    for (auto& mesh: results) {
        serializer.serialize(mesh.mesh_name);
        serializer.serialize(mesh.vertices.size());
        for (const auto& vertex: mesh.vertices) {
            serialize(serializer, vertex);
        }
        // serializer.serialize(std::span{mesh.indices});
    }
    gen_info("mesh", "Loaded {} meshes", results.size());

    if (physics_data) {
        export_physics_mesh(results, *physics_data, physics_type, *physx_state);
    }


    gen_info(__FUNCTION__, "loaded {} animations", animations.size());

    serializer.serialize(magic::anim);

    u64 anim_size = sizeof(utl::anim::animation_t) * animations.size();
    std::span<u8> anim_data{(u8*)animations.data(), anim_size};
    serializer.serialize((u64)animations.size());
    serializer.serialize(anim_data);

    std::span<u8> skeleton_data{(u8*)&skeleton, sizeof(skeleton)};
    serializer.serialize(skeleton_data);

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
        if (has_extension(file_name, "obj") || 
            has_extension(file_name, "fbx") ||
            has_extension(file_name, "glb") ||
            has_extension(file_name, "gltf")
        ) {

            bool make_physics = false;
            PhysicsColliderType collider = PhysicsColliderType::CONVEX;

            if (std::find(physx_config.convex.begin(), physx_config.convex.end(), file_name) != physx_config.convex.end()) {
                make_physics = true;
            }
            if (std::find(physx_config.trimeshes.begin(), physx_config.trimeshes.end(), file_name) != physx_config.trimeshes.end()) {
                collider = PhysicsColliderType::TRIMESH;
                make_physics = true;
            }

            std::vector<u8> physics_data;
            std::vector<u8> data = export_mesh<gfx::skinned_vertex_t>(
                arena, file_name, 
                make_physics ? &physics_data : 0,
                collider,
                make_physics ? &physx_state : 0
            );
            if (data.empty()) {
                physics_data.clear();
                data = export_mesh<gfx::vertex_t>(
                    arena, file_name, 
                    make_physics ? &physics_data : 0,
                    collider,
                    make_physics ? &physx_state : 0
                );
                pack_file(packed_file, file_name, std::move(data), magic::mesh);
            } else {
                pack_file(packed_file, file_name, std::move(data), magic::skel);
            }

            if (make_physics && physics_data.size()) {
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
            gen_info("export", "Unknown file type: {}", file_name);
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
    } else {
        gen_info("args", "Invalid argument count - {} provided, expected 4", argc);
    }

    return 0;
}