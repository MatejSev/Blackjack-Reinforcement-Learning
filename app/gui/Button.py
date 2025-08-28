import pygame

class Button:
    def __init__(self, x, y, w, h, text='', color=(0,0,255), text_color=(255,255,255)):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color
        self.text = text
        self.text_color = text_color
        self.font = pygame.font.Font(None, 32)
        self.txt_surface = self.font.render(text, True, text_color)
        self.clicked = False

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        screen.blit(self.txt_surface, (self.rect.x + (self.rect.width - self.txt_surface.get_width()) // 2, self.rect.y + (self.rect.height - self.txt_surface.get_height()) // 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.clicked = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.clicked and self.rect.collidepoint(event.pos):
                self.clicked = False
                return False
        return True
