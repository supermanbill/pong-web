import asyncio
import sys
import math
import random
import pygame

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

# ── Constants ──────────────────────────────────────────────
WIDTH, HEIGHT   = 800, 600
PANEL_W         = 120
WINDOW_W        = WIDTH + PANEL_W
FPS             = 60
SAMPLE_RATE     = 44100

PADDLE_W, PADDLE_H = 12, 90
PADDLE_SPEED    = 6
AI_SPEED        = 4.2
AI_PREDICT      = 0.55

BALL_SIZE       = 12
BALL_SPEED_INIT = 5.0
BALL_SPEED_MAX  = 12.0
BALL_ACCEL      = 0.3

WIN_SCORE       = 7
SHAKE_FRAMES    = 6

BTN_W, BTN_H    = 72, 110
BTN_X           = WIDTH + (PANEL_W - BTN_W) // 2
BTN_UP_Y        = HEIGHT // 2 - BTN_H - 10
BTN_DOWN_Y      = HEIGHT // 2 + 10

# Colors
BLACK      = (0,   0,   0)
WHITE      = (255, 255, 255)
GRAY       = (60,  60,  60)
PANEL_BG   = (18,  18,  18)
PANEL_LINE = (55,  55,  55)
CYAN       = (50,  220, 220)
YELLOW     = (255, 230,  50)
GREEN      = (80,  255, 130)
RED        = (255,  80,  80)
BALL_COLOR = (255, 255,   0)
BTN_NORMAL   = (255, 255, 255,  55)
BTN_PRESSED  = (255, 255, 255, 160)
BTN_BORDER   = (200, 200, 200, 140)

SPARK_COLORS = [
    (255, 255, 200), (255, 240, 80),
    (255, 180,  30), (255, 120, 20),
    (255, 255, 255),
]


# ── Sound Synthesis ────────────────────────────────────────
def _arr(melody, wave='sine', vol=0.5, decay=5.0):
    chunks = []
    for freq, dur in melody:
        n = int(dur * SAMPLE_RATE)
        t = np.linspace(0, dur, n, endpoint=False)
        if wave == 'triangle':
            w = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
        elif wave == 'square':
            w = np.sign(np.sin(2 * np.pi * freq * t))
        else:
            w = np.sin(2 * np.pi * freq * t)
        env = np.exp(-decay * t / max(dur, 1e-9))
        chunks.append(w * env * vol)
    data = (np.concatenate(chunks) * 32767).astype(np.int16)
    return np.ascontiguousarray(np.column_stack([data, data]))


def build_sounds():
    keys = ('hit', 'win', 'lose', 'tick', 'go')
    s = {k: None for k in keys}
    if not _NUMPY:
        return s
    try:
        s['hit']  = pygame.sndarray.make_sound(
            _arr([(700, 0.07)], wave='sine',  vol=0.8, decay=12))
        s['tick'] = pygame.sndarray.make_sound(
            _arr([(350, 0.09)], wave='square', vol=0.35, decay=14))
        s['go']   = pygame.sndarray.make_sound(
            _arr([(523,0.06),(659,0.06),(784,0.22)], wave='sine', vol=0.6, decay=4))
        s['win']  = pygame.sndarray.make_sound(
            _arr([(523,0.10),(659,0.10),(784,0.10),(1047,0.50)],
                 wave='sine', vol=0.65, decay=3))
        s['lose'] = pygame.sndarray.make_sound(
            _arr([(440,0.14),(392,0.14),(349,0.14),(262,0.60)],
                 wave='sine', vol=0.5, decay=3))
    except Exception:
        pass
    return s


# ── Touch Button ───────────────────────────────────────────
class TouchButton:
    def __init__(self, rect: pygame.Rect, direction: int) -> None:
        self.rect      = rect
        self.direction = direction

    def is_active(self) -> bool:
        if pygame.mouse.get_pressed()[0]:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        active = self.is_active()
        bg = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        bg.fill(BTN_PRESSED if active else BTN_NORMAL)
        pygame.draw.rect(bg, BTN_BORDER, bg.get_rect(), 2, border_radius=14)
        surface.blit(bg, self.rect.topleft)
        cx, cy = self.rect.centerx, self.rect.centery
        hw, hh = 18, 14
        color = YELLOW if active else (180, 180, 180)
        if self.direction == -1:
            pts = [(cx, cy-hh), (cx-hw, cy+hh), (cx+hw, cy+hh)]
        else:
            pts = [(cx, cy+hh), (cx-hw, cy-hh), (cx+hw, cy-hh)]
        pygame.draw.polygon(surface, color, pts)
        if active:
            pygame.draw.polygon(surface, WHITE, pts, 2)


# ── Spark Particle ─────────────────────────────────────────
class Particle:
    def __init__(self, x: float, y: float, direction: int) -> None:
        angle      = random.uniform(-math.pi * 0.55, math.pi * 0.55)
        speed      = random.uniform(2.5, 8.0)
        self.x     = x
        self.y     = y
        self.vx    = math.cos(angle) * speed * direction
        self.vy    = math.sin(angle) * speed
        self.life  = 1.0
        self.decay = random.uniform(0.035, 0.075)
        self.color = random.choice(SPARK_COLORS)
        self.size  = random.randint(2, 5)

    def update(self) -> None:
        self.x   += self.vx
        self.y   += self.vy
        self.vy  += 0.25
        self.vx  *= 0.96
        self.life -= self.decay

    def draw(self, surface: pygame.Surface) -> None:
        if self.life <= 0:
            return
        size = max(1, int(self.size * self.life))
        r, g, b = self.color
        pygame.draw.circle(surface,
                           (int(r*self.life), int(g*self.life), int(b*self.life)),
                           (int(self.x), int(self.y)), size)

    @property
    def alive(self) -> bool:
        return self.life > 0


# ── Paddles ────────────────────────────────────────────────
class Paddle:
    def __init__(self, x: int, up_key: int, down_key: int) -> None:
        self.rect     = pygame.Rect(x, HEIGHT//2 - PADDLE_H//2, PADDLE_W, PADDLE_H)
        self.up_key   = up_key
        self.down_key = down_key
        self.score    = 0

    def move(self, keys, touch_up=False, touch_down=False) -> None:
        if keys[self.up_key]   or touch_up:
            self.rect.y -= PADDLE_SPEED
        if keys[self.down_key] or touch_down:
            self.rect.y += PADDLE_SPEED
        self.rect.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))

    def draw(self, surface: pygame.Surface, color) -> None:
        pygame.draw.rect(surface, color, self.rect, border_radius=4)


class AIPaddle:
    def __init__(self, x: int) -> None:
        self.rect  = pygame.Rect(x, HEIGHT//2 - PADDLE_H//2, PADDLE_W, PADDLE_H)
        self.score = 0

    def move(self, ball) -> None:
        if ball.vx < 0:
            time_to_reach = (self.rect.centerx - ball.x) / ball.vx
            predicted_y   = ball.y + ball.vy * time_to_reach * AI_PREDICT
            predicted_y   = max(0.0, min(float(HEIGHT), predicted_y))
            target_y      = ball.y + (predicted_y - ball.y) * AI_PREDICT
        else:
            target_y = HEIGHT / 2
        diff = target_y - self.rect.centery
        step = min(abs(diff), AI_SPEED) * (1 if diff > 0 else -1)
        self.rect.y += step
        self.rect.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))

    def draw(self, surface: pygame.Surface, color) -> None:
        pygame.draw.rect(surface, color, self.rect, border_radius=4)


# ── Ball ───────────────────────────────────────────────────
class Ball:
    def __init__(self) -> None:
        self.reset()

    def reset(self, direction: int = 1) -> None:
        self.x     = float(WIDTH  // 2)
        self.y     = float(HEIGHT // 2)
        self.speed = BALL_SPEED_INIT
        self.vx    = BALL_SPEED_INIT * direction
        self.vy    = BALL_SPEED_INIT * (1 if direction > 0 else -1)

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x) - BALL_SIZE//2,
                           int(self.y) - BALL_SIZE//2,
                           BALL_SIZE, BALL_SIZE)

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy
        if self.y - BALL_SIZE//2 <= 0:
            self.y  = BALL_SIZE // 2
            self.vy = abs(self.vy)
        elif self.y + BALL_SIZE//2 >= HEIGHT:
            self.y  = HEIGHT - BALL_SIZE // 2
            self.vy = -abs(self.vy)

    def hit_paddle(self, paddle) -> None:
        rel = (self.y - paddle.rect.centery) / (PADDLE_H / 2)
        rel = max(-1.0, min(1.0, rel))
        self.speed = min(self.speed + BALL_ACCEL, BALL_SPEED_MAX)
        self.vx    = self.speed * (-1 if self.vx > 0 else 1)
        self.vy    = self.speed * rel

    def draw(self, surface: pygame.Surface) -> None:
        r  = BALL_SIZE // 2
        cx, cy = int(self.x), int(self.y)
        for radius, alpha in [(r+8, 30), (r+4, 70), (r+1, 140)]:
            glow = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*BALL_COLOR, alpha), (radius, radius), radius)
            surface.blit(glow, (cx-radius, cy-radius))
        pygame.draw.circle(surface, BALL_COLOR, (cx, cy), r)
        pygame.draw.circle(surface, WHITE, (cx-2, cy-2), 2)


# ── Helpers ────────────────────────────────────────────────
def draw_center_line(surface: pygame.Surface) -> None:
    dash_h, gap = 16, 10
    x, y = WIDTH // 2 - 1, 0
    while y < HEIGHT:
        pygame.draw.rect(surface, GRAY, (x, y, 2, dash_h))
        y += dash_h + gap


def draw_panel(surface: pygame.Surface) -> None:
    pygame.draw.rect(surface, PANEL_BG, (WIDTH, 0, PANEL_W, HEIGHT))
    pygame.draw.line(surface, PANEL_LINE, (WIDTH, 0), (WIDTH, HEIGHT), 2)


# ── Main (async for pygbag) ────────────────────────────────
async def main() -> None:
    try:
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 512)
    except Exception:
        pass
    pygame.init()

    screen      = pygame.display.set_mode((WINDOW_W, HEIGHT))
    render_surf = pygame.Surface((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong")
    clock = pygame.time.Clock()

    font_count = pygame.font.SysFont("monospace", 120, bold=True)
    font_score = pygame.font.SysFont("monospace",  64, bold=True)
    font_label = pygame.font.SysFont("monospace",  22)
    font_go    = pygame.font.SysFont("monospace",  56, bold=True)
    font_sub   = pygame.font.SysFont("monospace",  26, bold=True)
    font_hint  = pygame.font.SysFont("monospace",  19)

    sounds = build_sounds()

    left  = AIPaddle(20)
    right = Paddle(WIDTH - 20 - PADDLE_W, pygame.K_UP, pygame.K_DOWN)
    ball  = Ball()

    btn_up   = TouchButton(pygame.Rect(BTN_X, BTN_UP_Y,   BTN_W, BTN_H), -1)
    btn_down = TouchButton(pygame.Rect(BTN_X, BTN_DOWN_Y, BTN_W, BTN_H), +1)

    state         = "countdown"
    cd_val        = 5
    cd_tick       = 1.0
    flash_timer   = 0.0
    particles     = []
    shake         = 0
    winner_text   = ""
    result_played = False

    if sounds['tick']:
        sounds['tick'].play()

    def do_replay():
        nonlocal state, cd_val, cd_tick, flash_timer
        nonlocal shake, winner_text, result_played
        left.score  = 0
        right.score = 0
        ball.reset(direction=1)
        particles.clear()
        state         = "countdown"
        cd_val        = 5
        cd_tick       = 1.0
        flash_timer   = 0.0
        shake         = 0
        winner_text   = ""
        result_played = False
        if sounds['tick']:
            sounds['tick'].play()

    while True:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if state == "gameover" and event.key == pygame.K_SPACE:
                    do_replay()
            if state == "gameover":
                if event.type in (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
                    do_replay()

        # ── State logic ──
        if state == "countdown":
            cd_tick -= dt
            if cd_tick <= 0.0:
                cd_val  -= 1
                cd_tick  = 1.0
                if cd_val >= 0 and sounds['tick']:
                    sounds['tick'].play()
                if cd_val < 0:
                    state       = "start_flash"
                    flash_timer = 0.75
                    if sounds['go']:
                        sounds['go'].play()

        elif state == "start_flash":
            flash_timer -= dt
            if flash_timer <= 0:
                state = "playing"

        elif state == "playing":
            keys       = pygame.key.get_pressed()
            touch_up   = btn_up.is_active()
            touch_down = btn_down.is_active()

            left.move(ball)
            right.move(keys, touch_up=touch_up, touch_down=touch_down)
            ball.update()

            def on_hit(direction: int):
                nonlocal shake
                shake = SHAKE_FRAMES
                if sounds['hit']:
                    ratio = (ball.speed - BALL_SPEED_INIT) / (BALL_SPEED_MAX - BALL_SPEED_INIT)
                    sounds['hit'].set_volume(max(0.25, min(1.0, 0.25 + 0.75 * ratio)))
                    sounds['hit'].play()
                for _ in range(14):
                    particles.append(Particle(ball.x, ball.y, direction))

            if ball.rect.colliderect(left.rect) and ball.vx < 0:
                ball.x = left.rect.right + BALL_SIZE // 2
                ball.hit_paddle(left)
                on_hit(1)

            if ball.rect.colliderect(right.rect) and ball.vx > 0:
                ball.x = right.rect.left - BALL_SIZE // 2
                ball.hit_paddle(right)
                on_hit(-1)

            if ball.x < 0:
                right.score += 1
                particles.clear()
                ball.reset(direction=1)
            elif ball.x > WIDTH:
                left.score += 1
                particles.clear()
                ball.reset(direction=-1)

            if left.score >= WIN_SCORE and not result_played:
                winner_text   = "AI WINS!"
                state         = "gameover"
                result_played = True
                if sounds['lose']:
                    sounds['lose'].play()
            elif right.score >= WIN_SCORE and not result_played:
                winner_text   = "YOU WIN!"
                state         = "gameover"
                result_played = True
                if sounds['win']:
                    sounds['win'].play()

            for p in particles:
                p.update()
            particles = [p for p in particles if p.alive]

        # ── Draw ──
        render_surf.fill(BLACK)
        draw_center_line(render_surf)

        left.draw(render_surf,  CYAN)
        right.draw(render_surf, YELLOW)

        if state in ("playing", "gameover"):
            ball.draw(render_surf)

        for p in particles:
            p.draw(render_surf)

        ls = font_score.render(str(left.score),  True, CYAN)
        rs = font_score.render(str(right.score), True, YELLOW)
        render_surf.blit(ls, (WIDTH//2 - 80 - ls.get_width(), 20))
        render_surf.blit(rs, (WIDTH//2 + 80, 20))

        lbl_ai  = font_label.render("AI",  True, CYAN)
        lbl_you = font_label.render("You", True, YELLOW)
        render_surf.blit(lbl_ai,  (WIDTH//2 - 80 - lbl_ai.get_width(), 88))
        render_surf.blit(lbl_you, (WIDTH//2 + 80, 88))

        # Countdown overlay
        if state == "countdown":
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            render_surf.blit(overlay, (0, 0))
            num_surf = font_count.render(str(max(0, cd_val)), True, WHITE)
            pulse    = min(1.0, cd_tick / 0.25)
            scale    = 1.0 + 0.25 * pulse
            sw = int(num_surf.get_width()  * scale)
            sh = int(num_surf.get_height() * scale)
            if sw > 0 and sh > 0:
                scaled = pygame.transform.scale(num_surf, (sw, sh))
                render_surf.blit(scaled, (WIDTH//2 - sw//2, HEIGHT//2 - sh//2))

        elif state == "start_flash":
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            render_surf.blit(overlay, (0, 0))
            go_surf = font_count.render("Start!", True, GREEN)
            render_surf.blit(go_surf,
                             (WIDTH//2 - go_surf.get_width()//2,
                              HEIGHT//2 - go_surf.get_height()//2))

        elif state == "gameover":
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 170))
            render_surf.blit(overlay, (0, 0))

            color = GREEN if winner_text == "YOU WIN!" else RED
            go  = font_go.render("GAME OVER", True, WHITE)
            ws  = font_sub.render(winner_text, True, color)
            rp  = font_sub.render("REPLAY  /  SPACE", True, YELLOW)
            ex  = font_hint.render("EXIT GAME  /  ESC", True, GRAY)
            tap = font_hint.render("( or tap anywhere to replay )", True, GRAY)

            render_surf.blit(go,  (WIDTH//2 - go.get_width()//2,  HEIGHT//2 - 100))
            render_surf.blit(ws,  (WIDTH//2 - ws.get_width()//2,  HEIGHT//2 -  30))
            render_surf.blit(rp,  (WIDTH//2 - rp.get_width()//2,  HEIGHT//2 +  30))
            render_surf.blit(ex,  (WIDTH//2 - ex.get_width()//2,  HEIGHT//2 +  74))
            render_surf.blit(tap, (WIDTH//2 - tap.get_width()//2, HEIGHT//2 + 104))

        # Blit with optional shake
        screen.fill(BLACK)
        if shake > 0:
            ox, oy = random.randint(-4, 4), random.randint(-4, 4)
            shake -= 1
        else:
            ox, oy = 0, 0
        screen.blit(render_surf, (ox, oy))

        draw_panel(screen)
        btn_up.draw(screen)
        btn_down.draw(screen)

        pygame.display.flip()
        await asyncio.sleep(0)   # yield to browser event loop


asyncio.run(main())
