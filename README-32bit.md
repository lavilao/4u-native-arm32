# The 32-bit (armeabi-v7a) build

This fork runs the native app on 32-bit Android phones: the Redmi 9A (Helio G25, a 32-bit
userspace, Vulkan 1.1) that the port was made for, and any armv7 phone with a Vulkan 1.1 driver.

```
ANDROID_ABI=armeabi-v7a sh scripts/build_sdl3.sh
ANDROID_ABI=armeabi-v7a sh scripts/build_apk.sh
```

or take the APK from this repository's Actions ("The Android APKs"), which builds both arm64-v8a
and armeabi-v7a. The on-phone builder works there too: run "The 32-bit builder's CPython" once
(first time only), which builds CPython for Android's arm with python.org's own recipe (python.org
publishes Android builds for aarch64 and x86_64 only) and puts it on the `builder-tools` release;
the APK workflow pins its SHA-256 into the APK. Zig's armv7l wheel (PyPI, static musl: it runs on
Android as the aarch64 one does) needs no such step.

## What differs from the arm64 build

The guest is 32-bit ARM either way — the recompiled C of the game's code is the same — so the port
is the host's side:

- **Guest memory** (`mh4n/native/include/mh4n/memory.h`): a 64-bit host reserves the guest's whole
  4 GiB address space; a 32-bit host has no 4 GiB to give, and reserves only the 1 GiB the 3DS
  kernel manages (the New 3DS limit) plus the 64 KiB guard. Nothing beyond that range can be
  mapped, and a stray guest address beyond it wraps the host's 32-bit pointer arithmetic to below
  the reservation and faults there, loudly.
- **Fibers** (`mh4n/port/src/fiber.cpp`, `fiber_arm32.S`): the same context switch for the AAPCS
  registers of a 32-bit thread (r4-r11, sp, lr, d8-d15). The icache needs no flush: the game's
  code is compiled ahead of time, not written at run time.
- **The code library's build** (`mh4n/tools/recomp/build.py`): the compiler target is
  `armv7a-linux-androideabi29` (clang) or `arm-linux-android -mcpu=cortex_a7+neon -mfpu=neon-vfpv4
  -mfloat-abi=softfp` (Zig 0.13, which has no armv7a triple and whose arm default predates ARMv7);
  the library check reads the ELF as Elf32/ARM, and Zig's compiler_rt rides along beside
  `-nostdlib`, so the u64 helpers of the host (`__aeabi_uldivmod` and friends) are in the library
  instead of imports. The phone's own architecture is picked from `uname` (a PC cross-building can
  set `MH4N_ANDROID_ARCH=arm`).
- **The builder's launchers** (`app_native/builder/toolrun.c`): the loader that runs Zig's static
  program in the app's address space reads a 32-bit ARM ELF, lays out the stack and the auxiliary
  vector with 32-bit words, and enters in ARM or Thumb mode as the entry says.
- **The GPU backend** (patch `24-vulkan-1.1-instance.diff` of the vendored plume): the instance
  asks for Vulkan 1.1 instead of 1.2 — a 1.1-only driver (the PowerVR GE8320) refuses an instance
  that asks for more. Everything above 1.1 was already an optional, probed capability (timeline
  semaphores, dynamic rendering and the rest; the SPIR-V is 1.3 for such a device), and newer
  drivers keep working under a 1.1 instance.
- **libadrenotools** builds for armeabi-v7a as well (a 32-bit Adreno phone can still load its own
  drivers); the bcenabler byte patch is AArch64-only data that only an Adreno would read.
- **crash reports** read the fault's pc and lr from the 32-bit frame (`arm_pc`, `arm_lr`).

## Limits

A 32-bit phone is a slow one: the first on-phone build of the game's code library takes what its
eight little cores give, and the work folder's ~800 MB is what it is. The 4 GiB flat reservation
of the arm64 build would not fit in a 32-bit address space — hence the 1 GiB one — but no 3DS
process maps anything beyond that range, so nothing is lost.

The signing key of the CI APKs (`android/mh4u.keystore`) is in the repository: it says only who
signed these builds, and anyone can read it — fine for a personal port, not for anything that
needs a key to be secret.
