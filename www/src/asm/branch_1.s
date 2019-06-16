    mov r0.x, lane_id
    mov r1.x, u(0)
    push_mask LOOP_END
LOOP_PROLOG:
    lt.u32 r0.y, r0.x, u(16)
    add.u32 r0.x, r0.x, u(1)
    mask_nz r0.y
LOOP_BEGIN:
    add.u32 r1.x, r1.x, u(1)
    jmp LOOP_PROLOG
LOOP_END:
    ret