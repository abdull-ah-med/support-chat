import { cn } from "cn"
import Image from "next/image"

type BrandMarkProps = {
  className?: string
  size?: number
  alt?: string
}

const BrandMark = ({ className, size = 28, alt = "SupportChat" }: BrandMarkProps) => (
  <Image
    src="/icon-watermark.png"
    alt={alt}
    width={size}
    height={size}
    className={cn("shrink-0 rounded-lg object-contain", className)}
  />
)

export { BrandMark }
