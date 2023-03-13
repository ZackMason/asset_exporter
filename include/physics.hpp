#pragma once

#include "PxPhysicsAPI.h"

#include "PxPhysics.h"

#include "PxScene.h"


namespace phys {
    
    #define PVD_HOST "127.0.0.1"
    #define PX_RELEASE(x)	if(x)	{ x->release(); x = NULL; }

    class error_callback_t : public physx::PxErrorCallback {
    public:
        virtual void reportError(physx::PxErrorCode::Enum code, const char* message, const char* file,
            int line)
        {
            gen_error("physx", "PhysX Error Callback: {}:{} -\n\t{}\n", file, line, message);
        }
    };


    struct physx_state_t {
        physx::PxFoundation* foundation{nullptr};
        physx::PxPvd* pvd{nullptr};
        physx::PxPhysics* physics{nullptr};
        physx::PxDefaultCpuDispatcher* dispatcher{nullptr};
        physx::PxCooking* cooking{nullptr};
        error_callback_t error_callback;
        physx::PxDefaultAllocator default_allocator;
    };

    inline void
    init_physx_state(
        physx_state_t& state
    ) {
        using namespace physx;
        state.foundation = PxCreateFoundation(PX_PHYSICS_VERSION, state.default_allocator, state.error_callback);

        assert(state.foundation);

        // state.pvd = PxCreatePvd(*state.foundation);
        // PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10000);
        // state.pvd->connect(*transport, PxPvdInstrumentationFlag::eALL);

        state.physics = PxCreatePhysics(PX_PHYSICS_VERSION, *state.foundation, PxTolerancesScale(), true, state.pvd);

        assert(state.physics);
        state.dispatcher = PxDefaultCpuDispatcherCreate(2);

        state.cooking = PxCreateCooking(PX_PHYSICS_VERSION, *state.foundation, PxCookingParams(state.physics->getTolerancesScale()));
        assert(state.cooking);
    }


};