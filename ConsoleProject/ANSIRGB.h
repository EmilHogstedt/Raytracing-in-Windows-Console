#pragma once

//Not allowed to change this code without making it GNU LGPL
#define R(c) (((c) >> 16) & 0xff)
#define G(c) (((c) >>  8) & 0xff)
#define B(c) ( (c)        & 0xff)

#define CUBE_THRESHOLDS(a, b, c, d, e)		\
	if      (v < a) return IDX(0,   0);	\
	else if (v < b) return IDX(1,  95);	\
	else if (v < c) return IDX(2, 135);	\
	else if (v < d) return IDX(3, 175);	\
	else if (v < e) return IDX(4, 215);	\
	else            return IDX(5, 255);

#define IDX(i, v) ((((uint32_t)i * 36 + 16) << 24) | ((uint32_t)v << 16))

__device__ static uint32_t cube_index_red(uint8_t v) {
	CUBE_THRESHOLDS(38, 115, 155, 196, 235);
}

#undef IDX
#define IDX(i, v) ((((uint32_t)i * 6) << 24) | ((uint32_t)v << 8))

__device__ static uint32_t cube_index_green(uint8_t v) {
	CUBE_THRESHOLDS(36, 116, 154, 195, 235);
}

#undef IDX
#define IDX(i, v) (((uint32_t)i << 24) | (uint32_t)v)

__device__ static uint32_t cube_index_blue(uint8_t v) {
	CUBE_THRESHOLDS(35, 115, 155, 195, 235);
}

#undef IDX
#undef CUBE_THRESHOLDS

__constant__ uint32_t colours[256] = {
	/* The 16 system colours as used by default by xterm.  Taken
	   from XTerm-col.ad distributed with xterm source code. */
	0x000000, 0xcd0000, 0x00cd00, 0xcdcd00,
	0x0000ee, 0xcd00cd, 0x00cdcd, 0xe5e5e5,
	0x7f7f7f, 0xff0000, 0x00ff00, 0xffff00,
	0x5c5cff, 0xff00ff, 0x00ffff, 0xffffff,

	/* 6󬝲 cube.  One each axis, the six indices map to [0, 95,
	   135, 175, 215, 255] RGB component values. */
	0x000000, 0x00005f, 0x000087, 0x0000af,
	0x0000d7, 0x0000ff, 0x005f00, 0x005f5f,
	0x005f87, 0x005faf, 0x005fd7, 0x005fff,
	0x008700, 0x00875f, 0x008787, 0x0087af,
	0x0087d7, 0x0087ff, 0x00af00, 0x00af5f,
	0x00af87, 0x00afaf, 0x00afd7, 0x00afff,
	0x00d700, 0x00d75f, 0x00d787, 0x00d7af,
	0x00d7d7, 0x00d7ff, 0x00ff00, 0x00ff5f,
	0x00ff87, 0x00ffaf, 0x00ffd7, 0x00ffff,
	0x5f0000, 0x5f005f, 0x5f0087, 0x5f00af,
	0x5f00d7, 0x5f00ff, 0x5f5f00, 0x5f5f5f,
	0x5f5f87, 0x5f5faf, 0x5f5fd7, 0x5f5fff,
	0x5f8700, 0x5f875f, 0x5f8787, 0x5f87af,
	0x5f87d7, 0x5f87ff, 0x5faf00, 0x5faf5f,
	0x5faf87, 0x5fafaf, 0x5fafd7, 0x5fafff,
	0x5fd700, 0x5fd75f, 0x5fd787, 0x5fd7af,
	0x5fd7d7, 0x5fd7ff, 0x5fff00, 0x5fff5f,
	0x5fff87, 0x5fffaf, 0x5fffd7, 0x5fffff,
	0x870000, 0x87005f, 0x870087, 0x8700af,
	0x8700d7, 0x8700ff, 0x875f00, 0x875f5f,
	0x875f87, 0x875faf, 0x875fd7, 0x875fff,
	0x878700, 0x87875f, 0x878787, 0x8787af,
	0x8787d7, 0x8787ff, 0x87af00, 0x87af5f,
	0x87af87, 0x87afaf, 0x87afd7, 0x87afff,
	0x87d700, 0x87d75f, 0x87d787, 0x87d7af,
	0x87d7d7, 0x87d7ff, 0x87ff00, 0x87ff5f,
	0x87ff87, 0x87ffaf, 0x87ffd7, 0x87ffff,
	0xaf0000, 0xaf005f, 0xaf0087, 0xaf00af,
	0xaf00d7, 0xaf00ff, 0xaf5f00, 0xaf5f5f,
	0xaf5f87, 0xaf5faf, 0xaf5fd7, 0xaf5fff,
	0xaf8700, 0xaf875f, 0xaf8787, 0xaf87af,
	0xaf87d7, 0xaf87ff, 0xafaf00, 0xafaf5f,
	0xafaf87, 0xafafaf, 0xafafd7, 0xafafff,
	0xafd700, 0xafd75f, 0xafd787, 0xafd7af,
	0xafd7d7, 0xafd7ff, 0xafff00, 0xafff5f,
	0xafff87, 0xafffaf, 0xafffd7, 0xafffff,
	0xd70000, 0xd7005f, 0xd70087, 0xd700af,
	0xd700d7, 0xd700ff, 0xd75f00, 0xd75f5f,
	0xd75f87, 0xd75faf, 0xd75fd7, 0xd75fff,
	0xd78700, 0xd7875f, 0xd78787, 0xd787af,
	0xd787d7, 0xd787ff, 0xd7af00, 0xd7af5f,
	0xd7af87, 0xd7afaf, 0xd7afd7, 0xd7afff,
	0xd7d700, 0xd7d75f, 0xd7d787, 0xd7d7af,
	0xd7d7d7, 0xd7d7ff, 0xd7ff00, 0xd7ff5f,
	0xd7ff87, 0xd7ffaf, 0xd7ffd7, 0xd7ffff,
	0xff0000, 0xff005f, 0xff0087, 0xff00af,
	0xff00d7, 0xff00ff, 0xff5f00, 0xff5f5f,
	0xff5f87, 0xff5faf, 0xff5fd7, 0xff5fff,
	0xff8700, 0xff875f, 0xff8787, 0xff87af,
	0xff87d7, 0xff87ff, 0xffaf00, 0xffaf5f,
	0xffaf87, 0xffafaf, 0xffafd7, 0xffafff,
	0xffd700, 0xffd75f, 0xffd787, 0xffd7af,
	0xffd7d7, 0xffd7ff, 0xffff00, 0xffff5f,
	0xffff87, 0xffffaf, 0xffffd7, 0xffffff,

	/* Greyscale ramp.  This is calculated as (index - 232) * 10 + 8
	   repeated for each RGB component. */
	0x080808, 0x121212, 0x1c1c1c, 0x262626,
	0x303030, 0x3a3a3a, 0x444444, 0x4e4e4e,
	0x585858, 0x626262, 0x6c6c6c, 0x767676,
	0x808080, 0x8a8a8a, 0x949494, 0x9e9e9e,
	0xa8a8a8, 0xb2b2b2, 0xbcbcbc, 0xc6c6c6,
	0xd0d0d0, 0xdadada, 0xe4e4e4, 0xeeeeee,
};

__device__ uint32_t rgb_from_ansi256(uint8_t index) {
	return colours[index];
}

__device__ static uint32_t distance(uint32_t x, uint32_t y) {
	int32_t r_sum = R(x) + R(y);
	int32_t r = (int32_t)R(x) - (int32_t)R(y);
	int32_t g = (int32_t)G(x) - (int32_t)G(y);
	int32_t b = (int32_t)B(x) - (int32_t)B(y);
	return (1024 + r_sum) * r * r + 2048 * g * g + (1534 - r_sum) * b * b;
}

__device__ static uint8_t luminance(uint32_t rgb) {

	const uint32_t v = (UINT32_C(3567664) * R(rgb) +
		UINT32_C(11998547) * G(rgb) +
		UINT32_C(1211005) * B(rgb));

	//The first option out of these 2 is 5 times faster but less accurate.
	return (v + (UINT32_C(1) << 23)) >> 24;
	/*
	return sqrtf((float)R(rgb) * (float)R(rgb) * 0.2126729f +
		(float)G(rgb) * (float)G(rgb) * 0.7151521f +
		(float)B(rgb) * (float)B(rgb) * 0.0721750);
	*/
}

__device__ uint8_t ansi256_from_rgb(uint32_t rgb) {

	static const uint8_t ansi256_from_grey[256] = {
		 16,  16,  16,  16,  16, 232, 232, 232,
		232, 232, 232, 232, 232, 232, 233, 233,
		233, 233, 233, 233, 233, 233, 233, 233,
		234, 234, 234, 234, 234, 234, 234, 234,
		234, 234, 235, 235, 235, 235, 235, 235,
		235, 235, 235, 235, 236, 236, 236, 236,
		236, 236, 236, 236, 236, 236, 237, 237,
		237, 237, 237, 237, 237, 237, 237, 237,
		238, 238, 238, 238, 238, 238, 238, 238,
		238, 238, 239, 239, 239, 239, 239, 239,
		239, 239, 239, 239, 240, 240, 240, 240,
		240, 240, 240, 240,  59,  59,  59,  59,
		 59, 241, 241, 241, 241, 241, 241, 241,
		242, 242, 242, 242, 242, 242, 242, 242,
		242, 242, 243, 243, 243, 243, 243, 243,
		243, 243, 243, 244, 244, 244, 244, 244,
		244, 244, 244, 244, 102, 102, 102, 102,
		102, 245, 245, 245, 245, 245, 245, 246,
		246, 246, 246, 246, 246, 246, 246, 246,
		246, 247, 247, 247, 247, 247, 247, 247,
		247, 247, 247, 248, 248, 248, 248, 248,
		248, 248, 248, 248, 145, 145, 145, 145,
		145, 249, 249, 249, 249, 249, 249, 250,
		250, 250, 250, 250, 250, 250, 250, 250,
		250, 251, 251, 251, 251, 251, 251, 251,
		251, 251, 251, 252, 252, 252, 252, 252,
		252, 252, 252, 252, 188, 188, 188, 188,
		188, 253, 253, 253, 253, 253, 253, 254,
		254, 254, 254, 254, 254, 254, 254, 254,
		254, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 231,
		231, 231, 231, 231, 231, 231, 231, 231,
	};

	/* First of, if it抯 shade of grey, we know exactly the best colour that
	   approximates it. */
	if (R(rgb) == G(rgb) && G(rgb) == B(rgb)) {
		return ansi256_from_grey[rgb & 0xff];
	}

	uint8_t grey_index = ansi256_from_grey[luminance(rgb)];
	uint32_t grey_distance = distance(rgb, rgb_from_ansi256(grey_index));
	uint32_t cube = cube_index_red(R(rgb)) + cube_index_green(G(rgb)) +
		cube_index_blue(B(rgb));
	return distance(rgb, cube) < grey_distance ? cube >> 24 : grey_index;
}