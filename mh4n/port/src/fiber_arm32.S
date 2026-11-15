// ARM32 (armeabi-v7a) fiber context switch for Android, the counterpart of fiber_arm64.S for the
// app's 32-bit build. Bionic has no ucontext routines, so this saves and restores the callee-saved
// registers of the AAPCS (r4-r11, sp and lr, and d8-d15, which are callee-saved whenever the FPU
// is used, as the NEON-capable NDK code here does) and returns into the other context. The layout
// must match ArmContext in fiber.cpp; the static_asserts there check the offsets.
//
//   void mh4u_fiber_switch(ArmContext* from, ArmContext* to);
//
// ARM mode (the library around it may be Thumb; bx handles either at the entry, whose address
// keeps the Thumb bit). r0 receives the new context's `arg` before the jump: on a context's first
// entry that is the argument of mh4u_fiber_entry; afterwards it is ignored.

    .syntax unified
    .arch armv7-a
    .fpu neon

    .text
    .arm
    .p2align 2
    .globl mh4u_fiber_switch
    .hidden mh4u_fiber_switch
    .type mh4u_fiber_switch, %function
mh4u_fiber_switch:
    stmia r0, {r4-r11}            // r4-r11: offsets 0-28
    str   lr, [r0, #32]
    str   sp, [r0, #36]
    add   r3, r0, #40
    vstmia r3, {d8-d15}           // d8-d15: offsets 40-103

    ldmia r1, {r4-r11}
    ldr   lr, [r1, #32]
    ldr   sp, [r1, #36]
    add   r3, r1, #40
    vldmia r3, {d8-d15}
    ldr   r0, [r1, #104]          // the context's arg (r0 at its first entry)
    bx    lr
    .size mh4u_fiber_switch, .-mh4u_fiber_switch

    .section .note.GNU-stack,"",%progbits
