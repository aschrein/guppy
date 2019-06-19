    mov r0.x, lane_id
    lt.u32 r0.y, r0.x, u(16)
    ; if (lane_id > 16) {
    br_push r0.y, ELSE, CONVERGE
THEN:
    ; // Do smth
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x

    pop_mask

    ; } else {
ELSE:
    ; // Do smth else
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x
    mov r0.x, r0.x

    pop_mask
    ; }
CONVERGE:
    ret