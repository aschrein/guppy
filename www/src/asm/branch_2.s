; uint lane_id = get_lane_id();
    mov r0.x, lane_id
; for (uint i = lane_id; i < 16; i++) {
    push_mask LOOP_END
LOOP_PROLOG:
    lt.u32 r0.y, r0.x, u(16)
    add.u32 r0.x, r0.x, u(1)
    mask_nz r0.y
LOOP_BEGIN:
    ; // Do smth
    jmp LOOP_PROLOG
LOOP_END:
    ; // }
    ret